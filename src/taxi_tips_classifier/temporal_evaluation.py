"""
Temporal evaluation utilities for taxi tip classification model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

from .model import TaxiTipClassifier
from .predict import predict_and_evaluate


class TemporalModelEvaluator:
    """
    Class for evaluating model performance across multiple time periods.
    """
    
    def __init__(self, model_path: Optional[str] = None, classifier: Optional[TaxiTipClassifier] = None):
        """
        Initialize the temporal evaluator.
        
        Args:
            model_path: Path to saved model (if loading from disk)
            classifier: Already trained TaxiTipClassifier instance
        """
        if classifier is not None:
            self.classifier = classifier
        elif model_path is not None:
            self.classifier = TaxiTipClassifier()
            self.classifier.load_model(model_path)
        else:
            raise ValueError("Either model_path or classifier must be provided")
        
        self.evaluation_results = {}
        self.monthly_data_urls = {
            "2020-01": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet",
            "2020-02": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet", 
            "2020-03": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-03.parquet",
            "2020-04": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet",
            "2020-05": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-05.parquet",
            "2020-06": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-06.parquet"
        }
    
    def evaluate_single_month(self, month_key: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate model performance on a single month.
        
        Args:
            month_key: Month identifier (e.g., "2020-02")
            sample_size: Optional sample size for faster evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        if month_key not in self.monthly_data_urls:
            raise ValueError(f"Month {month_key} not available. Available: {list(self.monthly_data_urls.keys())}")
        
        print(f"\n📊 Evaluating model on {month_key} data...")
        
        # Load and preprocess data
        data_url = self.monthly_data_urls[month_key]
        monthly_data = self.classifier.load_and_preprocess_data(data_url)
        
        # Sample data if requested
        if sample_size is not None and len(monthly_data) > sample_size:
            print(f"   📝 Sampling {sample_size} records from {len(monthly_data)} total")
            monthly_data = monthly_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Evaluate
        results = self.classifier.evaluate(monthly_data)
        
        # Add metadata
        results['month'] = month_key
        results['data_url'] = data_url
        results['total_samples'] = len(monthly_data)
        results['evaluation_date'] = datetime.now().isoformat()
        
        print(f"   ✅ F1-Score: {results['f1_score']:.4f} | Accuracy: {results['accuracy']:.4f} | Samples: {len(monthly_data):,}")
        
        return results
    
    def evaluate_multiple_months(self, 
                                months: List[str], 
                                sample_size: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model performance across multiple months.
        
        Args:
            months: List of month identifiers
            sample_size: Optional sample size for each month
            
        Returns:
            Dictionary with results for each month
        """
        print(f"🚀 Starting temporal evaluation for {len(months)} months...")
        
        results = {}
        for month in months:
            try:
                month_results = self.evaluate_single_month(month, sample_size)
                results[month] = month_results
                self.evaluation_results[month] = month_results
            except Exception as e:
                print(f"❌ Error evaluating {month}: {e}")
                continue
        
        print(f"\n✅ Completed evaluation for {len(results)} months")
        return results
    
    def create_results_table(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Create a summary table of evaluation results.
        
        Args:
            results: Evaluation results (uses stored results if None)
            
        Returns:
            DataFrame with summary statistics
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            raise ValueError("No evaluation results available. Run evaluate_multiple_months first.")
        
        # Extract key metrics for each month
        table_data = []
        for month, result in results.items():
            row = {
                'Mes': month,
                'Muestras': f"{result['total_samples']:,}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'Propinas_Altas': result['support']['positive_samples'],
                'Propinas_Bajas': result['support']['negative_samples'],
                'Porcentaje_Altas': f"{(result['support']['positive_samples'] / result['total_samples']) * 100:.1f}%"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Mes')  # Sort by month
        
        return df
    
    def plot_temporal_performance(self, 
                                 results: Optional[Dict[str, Dict[str, Any]]] = None,
                                 metrics: List[str] = None,
                                 figsize: Tuple[int, int] = (15, 10),
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualizations of temporal model performance.
        
        Args:
            results: Evaluation results (uses stored results if None)
            metrics: Metrics to plot (default: ['f1_score', 'accuracy', 'precision', 'recall'])
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            raise ValueError("No evaluation results available.")
        
        if metrics is None:
            metrics = ['f1_score', 'accuracy', 'precision', 'recall']
        
        # Prepare data for plotting
        months = sorted(results.keys())
        month_labels = [month.replace('2020-', 'Mar ') for month in months]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Evaluación Temporal del Modelo - NYC Taxi Tips Classifier', fontsize=16, fontweight='bold')
        
        # Plot 1: Metrics over time
        ax1 = axes[0, 0]
        for metric in metrics:
            values = [results[month][metric] for month in months]
            ax1.plot(month_labels, values, marker='o', linewidth=2, label=metric.replace('_', ' ').title())
        
        ax1.set_title('Métricas de Rendimiento por Mes')
        ax1.set_xlabel('Mes')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Sample sizes
        ax2 = axes[0, 1]
        sample_sizes = [results[month]['total_samples'] for month in months]
        bars = ax2.bar(month_labels, sample_sizes, color='skyblue', alpha=0.7)
        ax2.set_title('Número de Muestras por Mes')
        ax2.set_xlabel('Mes')
        ax2.set_ylabel('Muestras')
        
        # Add value labels on bars
        for bar, size in zip(bars, sample_sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sample_sizes)*0.01,
                    f'{size:,}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Class distribution over time
        ax3 = axes[1, 0]
        high_tip_percentages = []
        for month in months:
            total = results[month]['total_samples']
            high_tips = results[month]['support']['positive_samples']
            percentage = (high_tips / total) * 100
            high_tip_percentages.append(percentage)
        
        ax3.plot(month_labels, high_tip_percentages, marker='s', color='green', linewidth=2)
        ax3.set_title('Porcentaje de Propinas Altas por Mes')
        ax3.set_xlabel('Mes')
        ax3.set_ylabel('% Propinas Altas')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: F1-Score with confidence intervals (approximate)
        ax4 = axes[1, 1]
        f1_scores = [results[month]['f1_score'] for month in months]
        
        # Calculate approximate confidence intervals based on sample size
        conf_intervals = []
        for month in months:
            n = results[month]['total_samples']
            f1 = results[month]['f1_score']
            # Approximate CI (simplified)
            margin = 1.96 * np.sqrt(f1 * (1 - f1) / n) if n > 0 else 0
            conf_intervals.append(margin)
        
        ax4.errorbar(month_labels, f1_scores, yerr=conf_intervals, 
                    marker='o', linewidth=2, capsize=5, color='red')
        ax4.set_title('F1-Score con Intervalos de Confianza')
        ax4.set_xlabel('Mes')
        ax4.set_ylabel('F1-Score')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado en: {save_path}")
        
        return fig
    
    def analyze_performance_degradation(self, 
                                      results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze performance degradation patterns.
        
        Args:
            results: Evaluation results (uses stored results if None)
            
        Returns:
            Dictionary with degradation analysis
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            raise ValueError("No evaluation results available.")
        
        months = sorted(results.keys())
        if len(months) < 2:
            raise ValueError("Need at least 2 months for degradation analysis.")
        
        # Calculate performance changes
        baseline_month = months[0]  # Assume first month is baseline (training month)
        baseline_f1 = results[baseline_month]['f1_score']
        
        degradation_analysis = {
            'baseline_month': baseline_month,
            'baseline_f1_score': baseline_f1,
            'monthly_changes': {},
            'max_degradation': 0,
            'most_degraded_month': None,
            'average_degradation': 0,
            'trends': {}
        }
        
        degradations = []
        for month in months[1:]:  # Skip baseline month
            current_f1 = results[month]['f1_score']
            change = current_f1 - baseline_f1
            degradation = -change if change < 0 else 0  # Only negative changes
            
            degradation_analysis['monthly_changes'][month] = {
                'f1_score': current_f1,
                'change_from_baseline': change,
                'degradation': degradation,
                'percentage_change': (change / baseline_f1) * 100
            }
            
            degradations.append(degradation)
        
        if degradations:
            degradation_analysis['max_degradation'] = max(degradations)
            degradation_analysis['average_degradation'] = np.mean(degradations)
            
            # Find most degraded month
            max_deg_idx = degradations.index(max(degradations))
            degradation_analysis['most_degraded_month'] = months[1:][max_deg_idx]
        
        # Analyze trends
        f1_scores = [results[month]['f1_score'] for month in months]
        sample_sizes = [results[month]['total_samples'] for month in months]
        high_tip_rates = [results[month]['support']['positive_samples'] / results[month]['total_samples'] for month in months]
        
        degradation_analysis['trends'] = {
            'f1_trend': 'declining' if f1_scores[-1] < f1_scores[0] else 'stable/improving',
            'sample_size_trend': 'decreasing' if sample_sizes[-1] < sample_sizes[0] else 'stable/increasing',
            'high_tip_rate_trend': 'decreasing' if high_tip_rates[-1] < high_tip_rates[0] else 'stable/increasing',
            'f1_variance': np.var(f1_scores),
            'correlation_f1_sample_size': np.corrcoef(f1_scores, sample_sizes)[0, 1] if len(f1_scores) > 1 else 0,
            'correlation_f1_tip_rate': np.corrcoef(f1_scores, high_tip_rates)[0, 1] if len(f1_scores) > 1 else 0
        }
        
        return degradation_analysis
    
    def generate_temporal_report(self, 
                               months: List[str],
                               sample_size: Optional[int] = None,
                               save_plots: bool = True,
                               plots_dir: str = "temporal_analysis") -> Dict[str, Any]:
        """
        Generate a complete temporal evaluation report.
        
        Args:
            months: List of months to evaluate
            sample_size: Optional sample size for each month
            save_plots: Whether to save plots
            plots_dir: Directory to save plots
            
        Returns:
            Complete report dictionary
        """
        import os
        
        print("🎯 Generando Reporte de Evaluación Temporal")
        print("=" * 50)
        
        # 1. Evaluate multiple months
        results = self.evaluate_multiple_months(months, sample_size)
        
        # 2. Create results table
        results_table = self.create_results_table(results)
        
        # 3. Generate plots
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, "temporal_performance.png")
            fig = self.plot_temporal_performance(results, save_path=plot_path)
        else:
            fig = self.plot_temporal_performance(results)
        
        # 4. Analyze degradation
        degradation_analysis = self.analyze_performance_degradation(results)
        
        # 5. Compile report
        report = {
            'evaluation_summary': {
                'months_evaluated': len(results),
                'evaluation_date': datetime.now().isoformat(),
                'sample_size_per_month': sample_size,
                'model_info': self.classifier.summary()
            },
            'results_table': results_table,
            'detailed_results': results,
            'degradation_analysis': degradation_analysis,
            'plots_generated': save_plots,
            'recommendations': self._generate_recommendations(degradation_analysis, results_table)
        }
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _generate_recommendations(self, 
                                degradation_analysis: Dict[str, Any], 
                                results_table: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Check for significant degradation
        if degradation_analysis['max_degradation'] > 0.05:  # 5% degradation threshold
            recommendations.append(
                f"⚠️  DEGRADACIÓN SIGNIFICATIVA: El modelo muestra una caída máxima de "
                f"{degradation_analysis['max_degradation']:.3f} en F1-score "
                f"({degradation_analysis['most_degraded_month']}). "
                f"Considera reentrenar el modelo con datos más recientes."
            )
        
        # Check for trend patterns
        if degradation_analysis['trends']['f1_trend'] == 'declining':
            recommendations.append(
                "📉 TENDENCIA DECLINANTE: El rendimiento muestra una tendencia a la baja. "
                "Implementa monitoreo continuo y reentrenamiento periódico."
            )
        
        # Check variance
        if degradation_analysis['trends']['f1_variance'] > 0.001:  # High variance threshold
            recommendations.append(
                "📊 ALTA VARIABILIDAD: El rendimiento es inconsistente entre meses. "
                "Revisa los datos de entrada y considera técnicas de regularización."
            )
        
        # Check correlation with tip rates
        tip_correlation = degradation_analysis['trends']['correlation_f1_tip_rate']
        if abs(tip_correlation) > 0.5:
            direction = "positiva" if tip_correlation > 0 else "negativa"
            recommendations.append(
                f"💡 CORRELACIÓN CON PROPINAS: Correlación {direction} ({tip_correlation:.3f}) "
                f"entre F1-score y tasa de propinas altas. "
                f"El modelo es sensible a cambios en el comportamiento de propinas."
            )
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "✅ RENDIMIENTO ESTABLE: El modelo mantiene un rendimiento consistente. "
                "Continúa con monitoreo regular mensual."
            )
        
        recommendations.extend([
            "🔄 Implementa un pipeline de reentrenamiento automático mensual.",
            "📊 Establece alertas cuando F1-score < 0.7 o caída > 5%.",
            "🎯 Considera incorporar características estacionales.",
            "🧪 Experimenta con técnicas de domain adaptation para drift temporal."
        ])
        
        return recommendations
    
    def _print_report_summary(self, report: Dict[str, Any]) -> None:
        """Print a formatted summary of the temporal evaluation report."""
        print("\n" + "="*60)
        print("📋 RESUMEN DEL REPORTE DE EVALUACIÓN TEMPORAL")
        print("="*60)
        
        # Summary stats
        summary = report['evaluation_summary']
        print(f"📅 Fecha de evaluación: {summary['evaluation_date'][:19]}")
        print(f"📊 Meses evaluados: {summary['months_evaluated']}")
        print(f"🎯 Muestras por mes: {summary['sample_size_per_month'] or 'Todos los datos'}")
        
        # Performance summary
        table = report['results_table']
        print(f"\n📈 RENDIMIENTO POR MES:")
        print(table.to_string(index=False))
        
        # Degradation analysis
        deg = report['degradation_analysis']
        print(f"\n🔍 ANÁLISIS DE DEGRADACIÓN:")
        print(f"   Mes base: {deg['baseline_month']} (F1: {deg['baseline_f1_score']:.4f})")
        print(f"   Degradación máxima: {deg['max_degradation']:.4f}")
        print(f"   Mes más degradado: {deg['most_degraded_month']}")
        print(f"   Tendencia F1: {deg['trends']['f1_trend']}")
        
        # Recommendations
        print(f"\n💡 RECOMENDACIONES:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)


def evaluate_model_temporal_performance(model_path: str,
                                      months: List[str] = None,
                                      sample_size: Optional[int] = None,
                                      save_report: bool = True) -> Dict[str, Any]:
    """
    Convenience function for temporal model evaluation.
    
    Args:
        model_path: Path to saved model
        months: List of months to evaluate (default: Jan-Jun 2020)
        sample_size: Optional sample size for faster evaluation
        save_report: Whether to save detailed report
        
    Returns:
        Complete evaluation report
    """
    if months is None:
        months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]
    
    evaluator = TemporalModelEvaluator(model_path=model_path)
    report = evaluator.generate_temporal_report(months, sample_size, save_plots=True)
    
    if save_report:
        report_path = "temporal_evaluation_report.joblib"
        joblib.dump(report, report_path)
        print(f"📄 Reporte completo guardado en: {report_path}")
    
    return report
