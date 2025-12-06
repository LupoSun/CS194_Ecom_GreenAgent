"""
Integration tests for the Green Agent
Tests the full agent workflow end-to-end
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from green_main_A2A import EcomGreenAgentExecutor, split_user_orders, henry_build_prompt
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue


class TestGreenAgentIntegration:
    """Integration tests for Green Agent executor"""
    
    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets for testing"""
        products = pd.DataFrame({
            "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "product_name": ["Apples", "Bananas", "Milk", "Bread", "Eggs", 
                           "Chicken", "Rice", "Pasta", "Cheese", "Yogurt"],
            "aisle_id": [1, 1, 2, 3, 2, 4, 5, 5, 2, 2],
            "aisle": ["fruits", "fruits", "dairy", "bakery", "dairy",
                     "meat", "grains", "grains", "dairy", "dairy"],
            "department_id": [1, 1, 2, 3, 2, 4, 5, 5, 2, 2],
            "department": ["produce", "produce", "dairy eggs", "bakery", "dairy eggs",
                         "meat seafood", "pantry", "pantry", "dairy eggs", "dairy eggs"]
        })
        
        # Create orders for 5 users with sufficient order sizes
        orders = []
        for user in range(1, 6):
            # Each user has 3 orders with 10-15 items each
            for order_num in range(1, 4):
                # Add 10-15 random products per order
                num_products = 10 + (user * order_num % 6)
                for i in range(num_products):
                    product_id = ((user + order_num + i) % 10) + 1
                    orders.append({
                        "user_id": user,
                        "order_number": order_num,
                        "order_id": 100 * user + order_num,
                        "product_id": product_id,
                        "add_to_cart_order": i + 1,
                        "days_since_prior_order": 7.0 if order_num > 1 else 0.0
                    })
        
        orders_df = pd.DataFrame(orders)
        return products, orders_df
    
    @pytest.fixture
    def executor(self, sample_datasets):
        """Create an executor with sample data"""
        products, orders = sample_datasets
        return EcomGreenAgentExecutor(products, orders)
    
    def test_executor_initialization(self, executor, sample_datasets):
        """Test that executor initializes correctly"""
        products, orders = sample_datasets
        
        assert executor.df_products is not None
        assert executor.df_orders is not None
        assert len(executor.df_products) == 10
        assert len(executor.runs) == 0
    
    def test_sample_user_ids_deterministic(self, executor):
        """Test that sampling with seed is deterministic"""
        sample1 = executor._sample_user_ids(3, random_state=42)
        sample2 = executor._sample_user_ids(3, random_state=42)
        
        assert sample1 == sample2
        assert len(sample1) == 3
    
    def test_sample_user_ids_different_seeds(self, executor):
        """Test that different seeds produce different samples"""
        sample1 = executor._sample_user_ids(3, random_state=42)
        sample2 = executor._sample_user_ids(3, random_state=123)
        
        # High probability they're different
        assert sample1 != sample2
    
    def test_sample_user_ids_too_many(self, executor):
        """Test error when requesting more users than available"""
        with pytest.raises(ValueError, match="exceeds unique user count"):
            executor._sample_user_ids(100)  # Only 5 users available
    
    def test_get_user_order_size(self, executor):
        """Test checking user order sizes"""
        # Test a valid user
        is_valid, size = executor._get_user_order_size(1, min_items=5)
        
        assert isinstance(is_valid, bool)
        assert isinstance(size, int)
        assert size > 0
    
    def test_sample_valid_user(self, executor):
        """Test sampling a user with minimum order size"""
        used_users = set()
        all_users = [1, 2, 3, 4, 5]
        
        # Sample a user with at least 5 items
        user_id = executor._sample_valid_user(all_users, used_users, min_items=5)
        
        assert user_id is not None
        assert user_id in all_users
        assert user_id not in used_users
    
    def test_sample_valid_user_excludes_used(self, executor):
        """Test that used users are excluded from sampling"""
        used_users = {1, 2, 3}
        all_users = [1, 2, 3, 4, 5]
        
        user_id = executor._sample_valid_user(all_users, used_users, min_items=5)
        
        assert user_id is not None
        assert user_id in [4, 5]  # Only available users
    
    def test_baseline_policy(self, executor, sample_datasets):
        """Test baseline policy (repeat previous order)"""
        products, orders = sample_datasets
        
        # Get previous order for user 1
        parts = split_user_orders(1, products, orders)
        basket = executor._baseline_policy(parts["previous_order_df"])
        
        assert isinstance(basket, dict)
        assert len(basket) > 0
        
        # All values should be positive integers (quantities)
        for product_id, qty in basket.items():
            assert isinstance(product_id, int)
            assert isinstance(qty, int)
            assert qty > 0
    
    def test_baseline_policy_empty_order(self, executor):
        """Test baseline with empty previous order"""
        empty_df = pd.DataFrame(columns=["product_id"])
        basket = executor._baseline_policy(empty_df)
        
        assert basket == {}
    
    def test_build_task_message(self, executor, sample_datasets):
        """Test building task message for white agent"""
        products, orders = sample_datasets
        executor.df_products = products
        executor.df_orders = orders
        
        task_info = {
            "user_id": 1,
            "railway_base_url": "http://test.com",
            "agent_key": "test_key"
        }
        
        message = executor._build_task_message(task_info)
        
        assert isinstance(message, str)
        assert "grocery shopping assistant" in message.lower()
        assert "test_key" in message
        assert "http://test.com" in message
        assert "##READY_FOR_CHECKOUT##" in message
        # Should NOT contain /checkout
        assert "/checkout" not in message.lower() or "do not call /checkout" in message.lower()
    
    def test_baseline_prompt_building(self, executor, sample_datasets):
        """Test prompt building for baseline evaluation"""
        products, orders = sample_datasets
        executor.df_products = products
        executor.df_orders = orders
        
        task_info = {"user_id": 1}
        prompt = executor._build_baseline_prompt(task_info)
        
        assert isinstance(prompt, str)
        assert "user 1" in prompt.lower()
        assert len(prompt) > 100  # Should be substantial
    
    def test_run_single_assessment_baseline_skip(self, executor, sample_datasets):
        """Test running a single baseline assessment (skipped - requires async support)"""
        pytest.skip("Async test - requires pytest-asyncio plugin")


class TestPromptQuality:
    """Tests to ensure prompts don't leak ground truth and are informative"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        products = pd.DataFrame({
            "product_id": [1, 2, 3, 4, 5],
            "product_name": ["Apple", "Banana", "Milk", "Bread", "Eggs"],
            "aisle_id": [1, 1, 2, 3, 2],
            "aisle": ["fruits", "fruits", "dairy", "bakery", "dairy"],
            "department_id": [1, 1, 2, 3, 2],
            "department": ["produce", "produce", "dairy eggs", "bakery", "dairy eggs"]
        })
        
        orders = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 1],
            "order_number": [1, 1, 2, 2, 3, 3],
            "order_id": [100, 100, 101, 101, 102, 102],
            "product_id": [1, 2, 3, 4, 1, 5],
            "days_since_prior_order": [0, 0, 7, 7, 5, 5]
        })
        
        return products, orders
    
    def test_prompt_no_ground_truth_leakage(self, sample_data):
        """CRITICAL: Ensure prompt doesn't reveal ground truth (current order)"""
        products, orders = sample_data
        
        # When predicting order 3, we should see orders 1 and 2 but NOT order 3
        parts = split_user_orders(1, products, orders)
        prompt = henry_build_prompt(
            parts["previous_orders_df"],
            parts["days_since_last"],
            1
        )
        
        # Current order (order 3) contains products 1 and 5
        # Product 1 (Apple) appears in history, so it's OK if it appears
        # Product 5 (Eggs) is NEW in order 3 and should NOT appear as "next order"
        
        # Check that we don't leak ground truth labels
        assert "next order contains" not in prompt.lower()
        assert "ground truth" not in prompt.lower()
        assert "will buy" not in prompt.lower()
        assert "correct answer" not in prompt.lower()
        
        # Verify the current order is NOT in the previous_orders_df
        assert 3 not in parts["previous_orders_df"]["order_number"].values
        
        # Verify only orders 1 and 2 are in history
        assert set(parts["previous_orders_df"]["order_number"].unique()) == {1, 2}
    
    def test_prompt_contains_useful_info(self, sample_data):
        """Test that prompt contains useful information for prediction"""
        products, orders = sample_data
        
        parts = split_user_orders(1, products, orders)
        prompt = henry_build_prompt(
            parts["previous_orders_df"],
            parts["days_since_last"],
            1
        )
        
        # Should contain key information
        assert "user 1" in prompt.lower() or "user" in prompt.lower()
        assert "days" in prompt.lower()
        
        # Should contain product names from history
        # Order 1 had Apple, Banana; Order 2 had Milk, Bread
        assert "apple" in prompt.lower() or "banana" in prompt.lower()
        
        # Should contain instructions
        assert "basket" in prompt.lower()
        assert "search" in prompt.lower() or "add" in prompt.lower()
    
    def test_prompt_does_not_prescribe_strategy(self, sample_data):
        """Test that prompt does NOT tell white agent how to shop"""
        products, orders = sample_data
        
        parts = split_user_orders(1, products, orders)
        prompt = henry_build_prompt(
            parts["previous_orders_df"],
            parts["days_since_last"],
            1
        )
        
        # Should NOT prescribe shopping strategy
        prompt_lower = prompt.lower()
        
        # These phrases should NOT be in the prompt
        forbidden_phrases = [
            "aim for a large basket",
            "add at least 15-20",
            "add all items from",
            "do not be shy"
        ]
        
        for phrase in forbidden_phrases:
            assert phrase not in prompt_lower, f"Prompt should not contain '{phrase}' - white agent should decide strategy"
        
        # Should still provide history data (not prescribe how to use it)
        assert "order" in prompt_lower  # Has order history
        assert "basket" in prompt_lower or "cart" in prompt_lower  # Mentions the task


class TestDataIntegrity:
    """Tests to ensure data splits are correct and no contamination"""
    
    def test_split_user_orders_no_future_leakage(self):
        """CRITICAL: Ensure split doesn't include future orders"""
        products = pd.DataFrame({
            "product_id": [1, 2, 3],
            "product_name": ["A", "B", "C"],
            "aisle_id": [1, 2, 3],
            "aisle": ["a1", "a2", "a3"],
            "department_id": [1, 2, 3],
            "department": ["d1", "d2", "d3"]
        })
        
        orders = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 1],
            "order_number": [1, 1, 2, 2, 3, 3],
            "order_id": [100, 100, 101, 101, 102, 102],
            "product_id": [1, 2, 1, 3, 2, 3],
            "days_since_prior_order": [0, 0, 5, 5, 3, 3]
        })
        
        # When predicting order 3 (n=3)
        parts = split_user_orders(1, products, orders)
        
        # Verify n is correct
        assert parts["n"] == 3
        
        # Current order should be exactly order 3
        assert set(parts["current_order_df"]["order_number"].unique()) == {3}
        
        # Previous orders should be ONLY orders < 3 (i.e., 1 and 2)
        prev_orders = set(parts["previous_orders_df"]["order_number"].unique())
        assert prev_orders == {1, 2}
        assert 3 not in prev_orders
        
        # Previous order (n-1) should be exactly order 2
        assert set(parts["previous_order_df"]["order_number"].unique()) == {2}
    
    def test_split_maintains_add_to_cart_order(self):
        """Test that order of items in cart is preserved"""
        products = pd.DataFrame({
            "product_id": [1, 2, 3],
            "product_name": ["A", "B", "C"],
            "aisle_id": [1, 2, 3],
            "aisle": ["a1", "a2", "a3"],
            "department_id": [1, 2, 3],
            "department": ["d1", "d2", "d3"]
        })
        
        orders = pd.DataFrame({
            "user_id": [1, 1, 1],
            "order_number": [1, 1, 1],
            "order_id": [100, 100, 100],
            "product_id": [3, 1, 2],
            "add_to_cart_order": [1, 2, 3],
            "days_since_prior_order": [0, 0, 0]
        })
        
        parts = split_user_orders(1, products, orders)
        
        # Check that items are sorted by add_to_cart_order
        cart_orders = parts["current_order_df"]["add_to_cart_order"].tolist()
        assert cart_orders == sorted(cart_orders)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
