"""
Unit tests for evaluate_basket function
Tests aisle/department scoring even when product_ids don't match
"""

import pytest
import pandas as pd
from green_main_A2A import evaluate_basket, _prf1


class TestPRF1:
    """Test the precision/recall/F1 helper function"""
    
    def test_perfect_match(self):
        """Test when truth and prediction are identical"""
        truth = {1, 2, 3}
        pred = {1, 2, 3}
        result = _prf1(truth, pred)
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["tp"] == 3
        assert result["fp"] == 0
        assert result["fn"] == 0
    
    def test_no_match(self):
        """Test when there's no overlap"""
        truth = {1, 2, 3}
        pred = {4, 5, 6}
        result = _prf1(truth, pred)
        
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["tp"] == 0
        assert result["fp"] == 3
        assert result["fn"] == 3
    
    def test_partial_match(self):
        """Test partial overlap"""
        truth = {1, 2, 3, 4}
        pred = {3, 4, 5, 6}
        result = _prf1(truth, pred)
        
        # TP=2 (3,4), FP=2 (5,6), FN=2 (1,2)
        assert result["tp"] == 2
        assert result["fp"] == 2
        assert result["fn"] == 2
        assert result["precision"] == 0.5  # 2/(2+2)
        assert result["recall"] == 0.5     # 2/(2+2)
        assert result["f1"] == 0.5
    
    def test_empty_sets(self):
        """Test with empty sets"""
        result = _prf1(set(), set())
        
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0


class TestEvaluateBasket:
    """Test the main evaluate_basket function"""
    
    @pytest.fixture
    def products_catalog(self):
        """Sample products catalog"""
        return pd.DataFrame({
            "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "product_name": [
                "Apples", "Bananas", "Milk", "Bread", "Eggs",
                "Chicken", "Rice", "Pasta", "Tomatoes", "Cheese"
            ],
            "aisle": [
                "fruits", "fruits", "dairy", "bakery", "dairy",
                "meat", "grains", "grains", "produce", "dairy"
            ],
            "department": [
                "produce", "produce", "dairy eggs", "bakery", "dairy eggs",
                "meat seafood", "pantry", "pantry", "produce", "dairy eggs"
            ]
        })
    
    @pytest.fixture
    def ground_truth_order(self):
        """Ground truth order (what user actually bought)"""
        return pd.DataFrame({
            "product_id": [1, 2, 3, 4, 5],  # Apples, Bananas, Milk, Bread, Eggs
            "aisle": ["fruits", "fruits", "dairy", "bakery", "dairy"],
            "department": ["produce", "produce", "dairy eggs", "bakery", "dairy eggs"]
        })
    
    def test_perfect_prediction(self, products_catalog, ground_truth_order):
        """Test when prediction exactly matches ground truth"""
        predicted = [1, 2, 3, 4, 5]  # Perfect match
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth_order,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # All metrics should be perfect
        assert result["products"]["precision"] == 1.0
        assert result["products"]["recall"] == 1.0
        assert result["products"]["f1"] == 1.0
        
        assert result["aisles"]["precision"] == 1.0
        assert result["aisles"]["recall"] == 1.0
        assert result["aisles"]["f1"] == 1.0
        
        assert result["departments"]["precision"] == 1.0
        assert result["departments"]["recall"] == 1.0
        assert result["departments"]["f1"] == 1.0
        
        assert result["blended_f1"] == 1.0
    
    def test_wrong_products_right_aisles(self, products_catalog, ground_truth_order):
        """
        CRITICAL TEST: Predicted products are WRONG, but from RIGHT aisles/departments
        This tests that we give credit for aisles/departments even when product_ids don't match
        """
        # Ground truth: [1=fruits/produce, 2=fruits/produce, 3=dairy/dairy eggs, 
        #                4=bakery/bakery, 5=dairy/dairy eggs]
        # Predict different products but same aisles/departments
        predicted = [1, 10, 6, 7]  
        # 1=fruits/produce (MATCH product, aisle, dept)
        # 10=dairy/dairy eggs (WRONG product, RIGHT dept, RIGHT aisle)
        # 6=meat/meat seafood (WRONG product, WRONG dept, WRONG aisle)
        # 7=grains/pantry (WRONG product, WRONG dept, WRONG aisle)
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth_order,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Product level: Only 1 match out of 5 ground truth, 4 predicted
        assert result["products"]["tp"] == 1  # Only product 1 matches
        assert result["products"]["fp"] == 3  # Products 10, 6, 7 are wrong
        assert result["products"]["fn"] == 4  # Products 2, 3, 4, 5 missed
        assert result["products"]["precision"] == 0.25  # 1/4
        assert result["products"]["recall"] == 0.2  # 1/5
        
        # Aisle level: Check which aisles are present
        # Truth aisles: {fruits, dairy, bakery} = 3 unique
        # Pred aisles: {fruits, dairy, meat, grains} = 4 unique
        # Overlap: {fruits, dairy} = 2
        assert result["aisles"]["tp"] == 2
        assert result["aisles"]["fp"] == 2  # meat, grains
        assert result["aisles"]["fn"] == 1  # bakery
        
        # Department level
        # Truth depts: {produce, dairy eggs, bakery} = 3 unique
        # Pred depts: {produce, dairy eggs, meat seafood, pantry} = 4 unique
        # Overlap: {produce, dairy eggs} = 2
        assert result["departments"]["tp"] == 2
        assert result["departments"]["fp"] == 2  # meat seafood, pantry
        assert result["departments"]["fn"] == 1  # bakery
        
        # Verify no missing mappings (all products found in catalog)
        assert result["meta"]["missing_mappings"]["pred_aisle_missing"] == 0
        assert result["meta"]["missing_mappings"]["pred_dept_missing"] == 0
    
    def test_products_not_in_ground_truth_but_in_catalog(self, products_catalog, ground_truth_order):
        """
        CRITICAL TEST: Predicted products NOT in ground truth, but ARE in catalog
        Should still get their aisles/departments mapped correctly
        """
        # Ground truth has products 1-5
        # Predict products 6-10 (all in catalog, none in ground truth)
        predicted = [6, 7, 8, 9, 10]
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth_order,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Product level: Zero matches
        assert result["products"]["tp"] == 0
        assert result["products"]["fp"] == 5
        assert result["products"]["fn"] == 5
        assert result["products"]["precision"] == 0.0
        assert result["products"]["recall"] == 0.0
        
        # Aisle/Department level: Should still be calculated!
        # Predicted products: 6=meat, 7=grains, 8=grains, 9=produce, 10=dairy
        # Predicted aisles: {meat, grains, produce, dairy} = 4 unique
        assert result["meta"]["pred_counts"]["aisles"] == 4
        
        # Predicted departments: {meat seafood, pantry, produce, dairy eggs} = 4 unique
        assert result["meta"]["pred_counts"]["departments"] == 4
        
        # Ground truth aisles: {fruits, dairy, bakery}
        # Predicted aisles: {meat, grains, produce, dairy}
        # Overlap: {dairy} = 1
        assert result["aisles"]["tp"] == 1
        
        # Ground truth departments: {produce, dairy eggs, bakery}
        # Predicted departments: {meat seafood, pantry, produce, dairy eggs}
        # Overlap: {produce, dairy eggs} = 2
        assert result["departments"]["tp"] == 2
        
        # CRITICAL: No missing mappings because all products are in catalog
        assert result["meta"]["missing_mappings"]["pred_aisle_missing"] == 0
        assert result["meta"]["missing_mappings"]["pred_dept_missing"] == 0
    
    def test_products_not_in_catalog(self, products_catalog, ground_truth_order):
        """Test when predicted products are not in catalog at all"""
        # Predict products 999, 1000 (not in catalog)
        predicted = [999, 1000, 1]  # 1 is in catalog, others aren't
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth_order,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Should track missing mappings
        assert result["meta"]["missing_mappings"]["pred_aisle_missing"] == 2
        assert result["meta"]["missing_mappings"]["pred_dept_missing"] == 2
        
        # Should still count the one product that was found
        assert result["meta"]["pred_counts"]["aisles"] == 1  # Only product 1's aisle
        assert result["meta"]["pred_counts"]["departments"] == 1
    
    def test_empty_prediction(self, products_catalog, ground_truth_order):
        """Test when no products are predicted"""
        predicted = []
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth_order,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Everything should be zero
        assert result["products"]["precision"] == 0.0
        assert result["products"]["recall"] == 0.0
        assert result["products"]["f1"] == 0.0
        assert result["blended_f1"] == 0.0
    
    def test_blended_f1_calculation(self, products_catalog, ground_truth_order):
        """Test that blended F1 is calculated correctly"""
        predicted = [1, 2]  # 2 out of 5 ground truth products
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth_order,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Manual calculation
        expected_blended = (
            0.6 * result["products"]["f1"] +
            0.2 * result["aisles"]["f1"] +
            0.2 * result["departments"]["f1"]
        )
        
        assert abs(result["blended_f1"] - expected_blended) < 1e-6
    
    def test_catalog_required(self, ground_truth_order):
        """Test that products_catalog is now required"""
        predicted = [1, 2, 3]
        
        # Should raise ValueError when catalog is None
        with pytest.raises(ValueError, match="products_catalog must be provided"):
            evaluate_basket(
                predicted_product_ids=predicted,
                nth_order_df=ground_truth_order,
                products_catalog=None,
                weights=(0.6, 0.2, 0.2)
            )
    
    def test_catalog_missing_columns(self, ground_truth_order):
        """Test that catalog must have required columns"""
        predicted = [1, 2, 3]
        
        # Catalog missing 'department' column
        incomplete_catalog = pd.DataFrame({
            "product_id": [1, 2, 3],
            "aisle": ["fruits", "dairy", "bakery"]
            # Missing 'department' column
        })
        
        with pytest.raises(ValueError, match="products_catalog must be provided"):
            evaluate_basket(
                predicted_product_ids=predicted,
                nth_order_df=ground_truth_order,
                products_catalog=incomplete_catalog,
                weights=(0.6, 0.2, 0.2)
            )


class TestEvaluateBasketEdgeCases:
    """Test edge cases and data quality issues"""
    
    def test_duplicate_products_in_prediction(self):
        """Test when prediction contains duplicate product_ids"""
        products_catalog = pd.DataFrame({
            "product_id": [1, 2, 3],
            "aisle": ["fruits", "dairy", "bakery"],
            "department": ["produce", "dairy eggs", "bakery"]
        })
        
        ground_truth = pd.DataFrame({
            "product_id": [1, 2],
            "aisle": ["fruits", "dairy"],
            "department": ["produce", "dairy eggs"]
        })
        
        # Predict with duplicates
        predicted = [1, 1, 1, 2]  # Product 1 appears 3 times
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Should deduplicate to {1, 2}
        assert result["products"]["tp"] == 2
        assert result["products"]["fp"] == 0
    
    def test_invalid_product_ids(self):
        """Test when prediction contains None or invalid IDs"""
        products_catalog = pd.DataFrame({
            "product_id": [1, 2, 3],
            "aisle": ["fruits", "dairy", "bakery"],
            "department": ["produce", "dairy eggs", "bakery"]
        })
        
        ground_truth = pd.DataFrame({
            "product_id": [1, 2],
            "aisle": ["fruits", "dairy"],
            "department": ["produce", "dairy eggs"]
        })
        
        # Include None and invalid types
        predicted = [1, None, 2]
        
        result = evaluate_basket(
            predicted_product_ids=predicted,
            nth_order_df=ground_truth,
            products_catalog=products_catalog,
            weights=(0.6, 0.2, 0.2)
        )
        
        # Should filter out None
        assert result["products"]["tp"] == 2
        assert result["products"]["fp"] == 0
    
    def test_weights_validation(self):
        """Test that weights must sum to 1.0"""
        products_catalog = pd.DataFrame({
            "product_id": [1],
            "aisle": ["fruits"],
            "department": ["produce"]
        })
        
        ground_truth = pd.DataFrame({
            "product_id": [1],
            "aisle": ["fruits"],
            "department": ["produce"]
        })
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            evaluate_basket(
                predicted_product_ids=[1],
                nth_order_df=ground_truth,
                products_catalog=products_catalog,
                weights=(0.5, 0.3, 0.1)  # Sum = 0.9
            )


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
