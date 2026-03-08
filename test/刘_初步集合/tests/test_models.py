"""
模型单元测试
验证各个模型模块的正确性
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.data_generator import MacroDataGenerator
from data.data_processor import DataProcessor
from models.midas.midas_model import MIDASModel
from models.dfm.dfm_model import DFMModel
from models.tslm.tslm_adapter import TSLMAdapter, TSLMConfig
from models.hybrid.hybrid_model import HybridPredictor, HybridModelConfig


class TestDataGenerator(unittest.TestCase):
    """测试数据生成器"""
    
    def setUp(self):
        self.generator = MacroDataGenerator()
    
    def test_gdp_generation(self):
        """测试GDP数据生成"""
        gdp = self.generator.generate_gdp()
        self.assertIsNotNone(gdp)
        self.assertGreater(len(gdp), 0)
        self.assertIn('gdp_value', gdp.columns)
        print(f"✓ GDP数据生成: {len(gdp)} 条记录")
    
    def test_monthly_generation(self):
        """测试月度数据生成"""
        gdp = self.generator.generate_gdp()
        monthly = self.generator.generate_monthly_indicators(gdp)
        self.assertIsNotNone(monthly)
        self.assertGreater(len(monthly), 0)
        print(f"✓ 月度数据生成: {len(monthly)} 条记录")
    
    def test_all_data_generation(self):
        """测试完整数据生成"""
        data = self.generator.generate_all_data()
        self.assertIn('gdp', data)
        self.assertIn('monthly', data)
        self.assertIn('daily', data)
        self.assertIn('weekly', data)
        print("✓ 完整数据生成成功")


class TestDataProcessor(unittest.TestCase):
    """测试数据处理器"""
    
    def setUp(self):
        generator = MacroDataGenerator()
        self.data = generator.generate_all_data()
        self.processor = DataProcessor()
    
    def test_adf_test(self):
        """测试ADF检验"""
        series = self.data['gdp']['gdp_value']
        result = self.processor.adf_test(series)
        self.assertIn('adf_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('is_stationary', result)
        print(f"✓ ADF检验: p-value={result['p_value']:.4f}")
    
    def test_make_stationary(self):
        """测试平稳化处理"""
        series = self.data['gdp']['gdp_value']
        stationary, diff_count = self.processor.make_stationary(series)
        self.assertIsNotNone(stationary)
        self.assertGreaterEqual(diff_count, 0)
        print(f"✓ 平稳化处理: 差分{diff_count}次")
    
    def test_remove_outliers(self):
        """测试异常值处理"""
        series = self.data['monthly']['electricity'].copy()
        # 添加异常值
        series.iloc[10] = series.mean() + 10 * series.std()
        cleaned = self.processor.remove_outliers(series, method='iqr')
        self.assertIsNotNone(cleaned)
        self.assertEqual(len(cleaned), len(series))
        print("✓ 异常值处理成功")
    
    def test_preprocess_pipeline(self):
        """测试预处理流水线"""
        processed = self.processor.preprocess_pipeline(self.data)
        self.assertIn('gdp', processed)
        self.assertIn('monthly', processed)
        print("✓ 预处理流水线成功")


class TestMIDASModel(unittest.TestCase):
    """测试MIDAS模型"""
    
    def setUp(self):
        generator = MacroDataGenerator()
        data = generator.generate_all_data()
        processor = DataProcessor()
        processed = processor.preprocess_pipeline(data)
        
        self.gdp = processed['gdp']['gdp_value_clean']
        self.electricity = processed['monthly']['electricity']
        self.processor = processor
    
    def test_midas_fit(self):
        """测试MIDAS拟合"""
        gdp_aligned, elec_aligned = self.processor.align_frequencies(
            pd.DataFrame(self.gdp), pd.DataFrame(self.electricity), 'Q', 'M'
        )
        
        midas = MIDASModel(n_lags=12)
        midas.fit(gdp_aligned['gdp_value_clean'], elec_aligned['electricity'])
        
        self.assertIsNotNone(midas.params)
        self.assertIsNotNone(midas.weights)
        print(f"✓ MIDAS拟合: beta0={midas.params.beta0:.4f}")
    
    def test_midas_predict(self):
        """测试MIDAS预测"""
        gdp_aligned, elec_aligned = self.processor.align_frequencies(
            pd.DataFrame(self.gdp), pd.DataFrame(self.electricity), 'Q', 'M'
        )
        
        midas = MIDASModel(n_lags=12)
        midas.fit(gdp_aligned['gdp_value_clean'], elec_aligned['electricity'])
        
        prediction = midas.predict(elec_aligned['electricity'], horizon=1)
        self.assertEqual(len(prediction), 1)
        print(f"✓ MIDAS预测: {prediction[0]:.2f}")
    
    def test_midas_summary(self):
        """测试MIDAS摘要"""
        gdp_aligned, elec_aligned = self.processor.align_frequencies(
            pd.DataFrame(self.gdp), pd.DataFrame(self.electricity), 'Q', 'M'
        )
        
        midas = MIDASModel(n_lags=12)
        midas.fit(gdp_aligned['gdp_value_clean'], elec_aligned['electricity'])
        
        summary = midas.get_model_summary()
        self.assertIn('parameters', summary)
        self.assertIn('goodness_of_fit', summary)
        print(f"✓ MIDAS摘要: RMSE={summary['goodness_of_fit']['rmse']:.2f}")


class TestDFMModel(unittest.TestCase):
    """测试DFM模型"""
    
    def setUp(self):
        generator = MacroDataGenerator()
        data = generator.generate_all_data()
        processor = DataProcessor()
        self.processed = processor.preprocess_pipeline(data)
    
    def test_dfm_fit(self):
        """测试DFM拟合"""
        monthly_numeric = self.processed['monthly'].select_dtypes(include=[np.number])
        
        dfm = DFMModel(n_factors=3)
        dfm.fit(monthly_numeric)
        
        self.assertIsNotNone(dfm.factors)
        self.assertIsNotNone(dfm.loadings)
        print(f"✓ DFM拟合: 提取{dfm.params.n_factors}个因子")
    
    def test_dfm_factors(self):
        """测试DFM因子提取"""
        monthly_numeric = self.processed['monthly'].select_dtypes(include=[np.number])
        
        dfm = DFMModel(n_factors=3)
        dfm.fit(monthly_numeric)
        
        factors_df = dfm.get_factor_df()
        self.assertEqual(factors_df.shape[1], 3)
        print(f"✓ DFM因子: {factors_df.shape}")
    
    def test_dfm_summary(self):
        """测试DFM摘要"""
        monthly_numeric = self.processed['monthly'].select_dtypes(include=[np.number])
        
        dfm = DFMModel(n_factors=3)
        dfm.fit(monthly_numeric)
        
        summary = dfm.get_model_summary()
        self.assertIn('explained_variance_ratio', summary)
        print(f"✓ DFM摘要: 解释方差={sum(summary['explained_variance_ratio']):.2%}")


class TestTSLMAdapter(unittest.TestCase):
    """测试TSLM适配器"""
    
    def test_tslm_initialization(self):
        """测试TSLM初始化"""
        config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
        tslm = TSLMAdapter(config)
        tslm.initialize(use_mock=True)
        
        self.assertTrue(tslm.is_initialized)
        print("✓ TSLM初始化成功")
    
    def test_tslm_forecast(self):
        """测试TSLM预测"""
        config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
        tslm = TSLMAdapter(config)
        tslm.initialize(use_mock=True)
        
        # 生成测试数据
        np.random.seed(42)
        test_series = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
        
        forecast = tslm.forecast(test_series, prediction_length=12)
        self.assertEqual(len(forecast), 12)
        print(f"✓ TSLM预测: {forecast[:3]}...")


class TestHybridModel(unittest.TestCase):
    """测试混合模型"""
    
    def setUp(self):
        generator = MacroDataGenerator()
        data = generator.generate_all_data()
        processor = DataProcessor()
        processed = processor.preprocess_pipeline(data)
        
        self.gdp = processed['gdp']['gdp_value_clean']
        self.monthly = processed['monthly']
        self.processor = processor
    
    def test_hybrid_fit(self):
        """测试混合模型拟合"""
        gdp_aligned, monthly_aligned = self.processor.align_frequencies(
            pd.DataFrame(self.gdp), self.monthly, 'Q', 'M'
        )
        
        # 创建子模型
        midas = MIDASModel(n_lags=12)
        if 'electricity' in monthly_aligned.columns:
            midas.fit(gdp_aligned['gdp_value_clean'], monthly_aligned['electricity'])
        
        config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
        tslm = TSLMAdapter(config)
        tslm.initialize(use_mock=True)
        
        # 创建混合模型
        hybrid_config = HybridModelConfig(linear_weight=0.6, nonlinear_weight=0.4)
        hybrid = HybridPredictor(hybrid_config)
        hybrid.set_models(midas, tslm)
        hybrid.fit(gdp_aligned['gdp_value_clean'])
        
        self.assertTrue(hybrid.fitted)
        print("✓ 混合模型拟合成功")
    
    def test_hybrid_predict(self):
        """测试混合模型预测"""
        gdp_aligned, monthly_aligned = self.processor.align_frequencies(
            pd.DataFrame(self.gdp), self.monthly, 'Q', 'M'
        )
        
        # 创建子模型
        midas = MIDASModel(n_lags=12)
        if 'electricity' in monthly_aligned.columns:
            midas.fit(gdp_aligned['gdp_value_clean'], monthly_aligned['electricity'])
        
        config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
        tslm = TSLMAdapter(config)
        tslm.initialize(use_mock=True)
        
        # 创建混合模型
        hybrid_config = HybridModelConfig(linear_weight=0.6, nonlinear_weight=0.4)
        hybrid = HybridPredictor(hybrid_config)
        hybrid.set_models(midas, tslm)
        hybrid.fit(gdp_aligned['gdp_value_clean'])

        prediction = hybrid.predict(prediction_length=1)
        self.assertIn('linear_prediction', prediction)
        self.assertIn('nonlinear_prediction', prediction)
        self.assertIn('final_prediction', prediction)
        print(f"✓ 混合模型预测: {prediction['final_prediction'][0]:.2f}")


def run_tests():
    """运行所有测试"""
    print("="*60)
    print("宏观经济预测系统 - 单元测试")
    print("="*60 + "\n")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMIDASModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDFMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestTSLMAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridModel))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    print(f"测试总数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
