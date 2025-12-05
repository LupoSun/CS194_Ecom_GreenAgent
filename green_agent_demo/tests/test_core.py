import unittest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path to ensure imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "green_agent_demo"))
sys.path.insert(0, str(project_root / "real_white_agent"))
sys.path.insert(0, str(project_root))

from my_white_agent import extract_context_from_message, ShopAPI
from green_agent_demo.green_main_A2A import (
    split_user_orders, 
    henry_build_prompt, 
    _prf1, 
    evaluate_basket, 
    EcomGreenAgentExecutor
)

class TestWhiteAgentHelpers(unittest.TestCase):
    def test_extract_context_json(self):
        msg = '{"agent_key": "k1", "environment_base": "http://test", "user_id": 123}'
        ctx = extract_context_from_message(msg)
        self.assertEqual(ctx["agent_key"], "k1")
        self.assertEqual(ctx["environment_base"], "http://test")
        self.assertEqual(ctx["user_id"], 123)

    def test_extract_context_text(self):
        msg = "Use agent_key: my-key-123 and Base URL: https://api.com for user 99"
        ctx = extract_context_from_message(msg)
        self.assertEqual(ctx["agent_key"], "my-key-123")
        self.assertEqual(ctx["environment_base"], "https://api.com")
        self.assertEqual(ctx["user_id"], "99")

    def test_extract_context_missing_agent_key(self):
        msg = "Just a message with user 99"
        ctx = extract_context_from_message(msg)
        self.assertIsNone(ctx["agent_key"])
        self.assertEqual(ctx["user_id"], "99")

    def test_extract_context_fallback_env(self):
        msg = "agent_key: k1 user: 1 on green-agent-production.up.railway.app"
        ctx = extract_context_from_message(msg)
        self.assertEqual(ctx["environment_base"], "https://green-agent-production.up.railway.app")

    @patch('my_white_agent.requests.Session')
    def test_shop_api_search(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [{"product_id": 1, "name": "Test"}]}
        mock_session.get.return_value = mock_response

        api = ShopAPI("http://base", "key1")
        res = api.search_products("query")
        
        self.assertIn("Test", res)
        mock_session.get.assert_called_with(
            "http://base/search", 
            params={"q": "query", "agent_key": "key1"}, 
            timeout=10
        )

    @patch('my_white_agent.requests.Session')
    def test_shop_api_add_to_cart(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_session.post.return_value = mock_response

        api = ShopAPI("http://base", "key1")
        res = api.add_to_cart(123, 2)
        
        self.assertIn("Successfully added", res)
        mock_session.post.assert_called_with(
            "http://base/cart/add", 
            json={"agent_key": "key1", "items": [{"product_id": 123, "qty": 2}]}, 
            timeout=10
        )

    @patch('my_white_agent.requests.Session')
    def test_shop_api_get_product(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"product_id": 123, "name": "Detail"}
        mock_session.get.return_value = mock_response

        api = ShopAPI("http://base", "key1")
        res = api.get_product(123)
        
        self.assertIn("Detail", res)
        mock_session.get.assert_called_with(
            "http://base/get_product", 
            params={"product_id": 123, "agent_key": "key1"}, 
            timeout=10
        )

    @patch('my_white_agent.requests.Session')
    def test_shop_api_view_cart(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_session.get.return_value = mock_response

        api = ShopAPI("http://base", "key1")
        res = api.view_cart()
        
        self.assertIn("items", res)
        mock_session.get.assert_called_with(
            "http://base/cart", 
            params={"agent_key": "key1"}, 
            timeout=10
        )

class TestGreenAgentCore(unittest.TestCase):
    def setUp(self):
        self.df_products = pd.DataFrame({
            "product_id": [1, 2, 3],
            "product_name": ["P1", "P2", "P3"],
            "aisle_id": [10, 10, 11],
            "department_id": [20, 20, 21],
            "aisle": ["A1", "A1", "A2"],
            "department": ["D1", "D1", "D2"]
        })
        self.df_orders = pd.DataFrame({
            "user_id": [1, 1, 1],
            "order_number": [1, 1, 2],
            "product_id": [1, 2, 1],
            "order_id": [100, 100, 101],
            "days_since_prior_order": [0, 0, 7]
        })
        self.executor = EcomGreenAgentExecutor(self.df_products, self.df_orders)

    def test_split_user_orders(self):
        parts = split_user_orders(1, self.df_products, self.df_orders)
        self.assertEqual(len(parts["current_order_df"]), 1)
        self.assertEqual(parts["current_order_df"].iloc[0]["product_id"], 1)
        self.assertEqual(len(parts["previous_orders_df"]), 2)
        self.assertEqual(parts["n"], 2)

    def test_henry_build_prompt(self):
        parts = split_user_orders(1, self.df_products, self.df_orders)
        prompt = henry_build_prompt(parts["previous_orders_df"], 7.0, 1)
        self.assertIn("grocery shopping assistant for user 1", prompt)
        self.assertIn("P1", prompt)
        self.assertIn("AIM FOR A LARGE BASKET", prompt)

    def test_prf1_perfect(self):
        metrics = _prf1({1, 2, 3}, {1, 2, 3})
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_prf1_no_match(self):
        metrics = _prf1({1, 2}, {3, 4})
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_prf1_partial(self):
        metrics = _prf1({1, 2}, {2, 3})
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["precision"], 0.5) # 1/2
        self.assertEqual(metrics["recall"], 0.5)    # 1/2
        self.assertEqual(metrics["f1"], 0.5)

    def test_evaluate_basket_blended(self):
        # Truth: {1} -> Aisle A1, Dept D1
        # Pred: {2} -> Aisle A1, Dept D1
        # PIDs don't match, but Aisle/Dept do match.
        truth_df = pd.DataFrame({"product_id": [1], "aisle": ["A1"], "department": ["D1"]})
        
        # We assume df_products has mappings for both 1 and 2
        res = evaluate_basket(
            predicted_product_ids=[2],
            nth_order_df=truth_df,
            products_catalog=self.df_products,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Product F1 should be 0.0
        self.assertEqual(res["products"]["f1"], 0.0)
        # Aisle F1 should be 1.0 (Both A1)
        self.assertEqual(res["aisles"]["f1"], 1.0)
        # Dept F1 should be 1.0 (Both D1)
        self.assertEqual(res["departments"]["f1"], 1.0)
        
        # Blended = 0.6*0 + 0.2*1 + 0.2*1 = 0.4
        self.assertAlmostEqual(res["blended_f1"], 0.4)

    def test_baseline_policy(self):
        # Previous order has items {1, 2} from setup
        prev_df = self.df_orders[self.df_orders["order_number"] == 1]
        basket = self.executor._baseline_policy(prev_df)
        
        self.assertEqual(basket[1], 1)
        self.assertEqual(basket[2], 1)
        self.assertEqual(len(basket), 2)

    def test_baseline_policy_empty(self):
        empty_df = pd.DataFrame(columns=["product_id"])
        basket = self.executor._baseline_policy(empty_df)
        self.assertEqual(basket, {})

    def test_sample_user_ids(self):
        # Only user 1 exists in setup
        sampled = self.executor._sample_user_ids(1, random_state=42)
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0], 1)
        
        with self.assertRaises(ValueError):
            self.executor._sample_user_ids(5)  # More than available

class TestContaminationAndAntiCheat(unittest.TestCase):
    def setUp(self):
        self.df_products = pd.DataFrame({
            "product_id": [1, 2, 3],
            "product_name": ["P1", "P2", "P3"],
            "aisle_id": [10, 10, 11],
            "department_id": [20, 20, 21],
            "aisle": ["A1", "A1", "A2"],
            "department": ["D1", "D1", "D2"]
        })
        self.df_orders = pd.DataFrame({
            "user_id": [1, 1, 1],
            "order_number": [1, 1, 2],
            "product_id": [1, 2, 1],
            "order_id": [100, 100, 101],
            "days_since_prior_order": [0, 0, 7]
        })
        self.executor = EcomGreenAgentExecutor(self.df_products, self.df_orders)

    def test_prompt_no_contamination(self):
        """
        Ensure the prompt NEVER contains the ground truth (current order) items.
        User 1 has two orders:
        - Order 1 (History): P1, P2
        - Order 2 (Target): P1 (Ground Truth)
        
        The prompt should mention Order 1 but MUST NOT mention Order 2's specific content
        as a 'future' purchase or ground truth.
        """
        parts = split_user_orders(1, self.df_products, self.df_orders)
        prompt = henry_build_prompt(parts["previous_orders_df"], 7.0, 1)
        
        # The prompt should contain info about Order 1 (P1, P2)
        # But crucially, it should not inadvertently leak the ground truth label
        # or the specific 'Order 2' content in a way that reveals it is the answer.
        # Since Order 2 contains P1, and P1 is also in history, P1 will appear.
        # But we must ensure the prompt doesn't say "Next order contains: P1".
        
        self.assertNotIn("Next order contains", prompt)
        self.assertNotIn("Ground truth", prompt)
        
        # Ensure we are not passing the current order dataframe to the prompt builder
        # This verifies the orchestration logic
        
    def test_split_logic_leakage(self):
        """
        Verify that split_user_orders strictly separates past from future.
        If we are predicting Order N, the history MUST NOT include Order N or N+1.
        """
        # Add a 3rd order to test N=2 logic properly
        df_orders_extended = self.df_orders.copy()
        new_row = pd.DataFrame([{
            "user_id": 1, "order_number": 3, "product_id": 3, 
            "order_id": 102, "days_since_prior_order": 5
        }])
        df_orders_extended = pd.concat([df_orders_extended, new_row], ignore_index=True)
        
        # Test for N=3 (predicting Order 3)
        parts = split_user_orders(1, self.df_products, df_orders_extended)
        
        # Current order should be #3
        self.assertEqual(parts["n"], 3)
        self.assertTrue((parts["current_order_df"]["order_number"] == 3).all())
        
        # Previous orders should ONLY be < 3 (so 1 and 2)
        history_nums = parts["previous_orders_df"]["order_number"].unique()
        self.assertIn(1, history_nums)
        self.assertIn(2, history_nums)
        self.assertNotIn(3, history_nums)
        
        # Ensure no future leakage (if there were order 4)
        
    def test_random_sampling_integrity(self):
        """
        Verify that sampling is deterministic with seed (anti-cheat: ensures we test same users)
        but different without seed or with different seed.
        """
        # Create dummy users 1-10
        users = []
        for i in range(1, 11):
            users.append({"user_id": i, "order_number": 1, "product_id": 1})
        df_many = pd.DataFrame(users)
        exec_many = EcomGreenAgentExecutor(self.df_products, df_many)
        
        # Same seed -> Same users
        s1 = exec_many._sample_user_ids(5, random_state=42)
        s2 = exec_many._sample_user_ids(5, random_state=42)
        self.assertEqual(s1, s2)
        
        # Different seed -> Different users (high probability)
        s3 = exec_many._sample_user_ids(5, random_state=1)
        self.assertNotEqual(s1, s3)

if __name__ == '__main__':
    unittest.main()
