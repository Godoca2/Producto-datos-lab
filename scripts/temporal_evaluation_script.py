#!/usr/bin/env python3
"""
Script para Evaluación Temporal del Modelo NYC Taxi Tips
========================================================

Este script ejecuta una evaluación completa del rendimiento del modelo
a lo largo de múltiples meses para demostrar y analizar el comportamiento
temporal del clasificador de propinas.

Uso:
    python temporal_evaluation_script.py
    
    # Con parámetros personalizados:
    python temporal_evaluation_script.py --months 2020-01 2020-02 2020-03 --sample-size 50000
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

from taxi_tips_classifier.temporal_evaluation import TemporalModelEvaluator, evaluate_model_temporal_performance
from taxi_tips_classifier.model import TaxiTipClassifier
import matplotlib.pyplot as plt


def main():
    """Función principal para ejecutar la evaluación temporal."""
    
    parser = argparse.ArgumentParser(
        description="Evaluación Temporal del Modelo NYC Taxi Tips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Evaluación básica con 6 meses
  python temporal_evaluation_script.py
  
  # Evaluación con meses específicos
  python temporal_evaluation_script.py --months 2020-01 2020-02 2020-03 2020-04
  
  # Evaluación rápida con muestras pequeñas
  python temporal_evaluation_script.py --sample-size 10000
  
  # Solo generar tabla sin gráficos
  python temporal_evaluation_script.py --no-plots
        """
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="random_forest.joblib",
        help="Ruta al modelo entrenado (default: random_forest.joblib)"
    )
    
    parser.add_argument(
        "--months", 
        nargs="+", 
        default=["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"],
        help="Meses a evaluar (formato: 2020-01). Default: Enero-Junio 2020"
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None,
        help="Número de muestras por mes (default: todos los datos)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="temporal_analysis_results",
        help="Directorio para guardar resultados (default: temporal_analysis_results)"
    )
    
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="No generar gráficos, solo tabla de resultados"
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Evaluación rápida con 3 meses y muestras pequeñas"
    )
    
    args = parser.parse_args()
    
    # Configuración rápida
    if args.quick:
        args.months = ["2020-01", "2020-02", "2020-03"]
        args.sample_size = 20000
        print("🚀 Modo rápido activado: 3 meses, 20k muestras por mes")
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Modelo no encontrado en {args.model_path}")
        print("💡 Asegúrate de haber entrenado un modelo primero.")
        print("   Puedes ejecutar: python scripts/train_simple.py")
        return 1
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎯 EVALUACIÓN TEMPORAL DEL MODELO NYC TAXI TIPS")
    print("=" * 60)
    print(f"📁 Modelo: {args.model_path}")
    print(f"📅 Meses: {', '.join(args.months)}")
    print(f"📊 Muestras por mes: {args.sample_size or 'Todos los datos'}")
    print(f"💾 Resultados en: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Inicializar evaluador
        evaluator = TemporalModelEvaluator(model_path=args.model_path)
        
        # Ejecutar evaluación completa
        report = evaluator.generate_temporal_report(
            months=args.months,
            sample_size=args.sample_size,
            save_plots=not args.no_plots,
            plots_dir=args.output_dir
        )
        
        # Guardar tabla de resultados
        results_table = report['results_table']
        table_path = os.path.join(args.output_dir, "resultados_temporales.csv")
        results_table.to_csv(table_path, index=False)
        print(f"📄 Tabla de resultados guardada: {table_path}")
        
        # Guardar reporte completo
        import joblib
        report_path = os.path.join(args.output_dir, "reporte_completo.joblib")
        joblib.dump(report, report_path)
        print(f"📄 Reporte completo guardado: {report_path}")
        
        # Análisis crítico
        print("\n" + "="*60)
        print("🔍 ANÁLISIS CRÍTICO Y RECOMENDACIONES")
        print("="*60)
        
        degradation = report['degradation_analysis']
        trends = degradation['trends']
        
        print(f"\n📊 Consistencia del Rendimiento:")
        print(f"   • Varianza F1-Score: {trends['f1_variance']:.6f}")
        print(f"   • Tendencia general: {trends['f1_trend']}")
        print(f"   • Degradación máxima: {degradation['max_degradation']:.4f}")
        
        print(f"\n🔗 Factores Correlacionados:")
        print(f"   • Correlación F1 vs Tamaño muestra: {trends['correlation_f1_sample_size']:.3f}")
        print(f"   • Correlación F1 vs Tasa propinas altas: {trends['correlation_f1_tip_rate']:.3f}")
        
        print(f"\n💡 Factores que Explican la Variación:")
        
        # Análisis de factores
        tip_correlation = trends['correlation_f1_tip_rate']
        if abs(tip_correlation) > 0.5:
            direction = "fuerte correlación positiva" if tip_correlation > 0 else "fuerte correlación negativa"
            print(f"   • {direction} con cambios en comportamiento de propinas")
        
        if trends['f1_variance'] > 0.001:
            print(f"   • Alta variabilidad sugiere sensibilidad a cambios estacionales")
        
        if degradation['max_degradation'] > 0.05:
            print(f"   • Degradación significativa indica concept drift temporal")
        
        print(f"\n🎯 Recomendaciones para Mejorar Robustez:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Mostrar gráficos si están habilitados
        if not args.no_plots:
            print(f"\n📊 Gráficos generados en: {args.output_dir}/temporal_performance.png")
            print("💡 Abre el archivo de imagen para ver las visualizaciones.")
        
        print(f"\n✅ Evaluación temporal completada exitosamente")
        print(f"📁 Todos los resultados están en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
