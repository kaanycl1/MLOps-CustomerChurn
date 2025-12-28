import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

def run_phase4_monitoring():
    # Dosya yolları
    ref_path = "data/raw/customer_churn_dataset-training-master.csv"
    curr_path = "artifacts/inference_logs.csv"
    report_path = "artifacts/drift_report.html"

    # Verileri yükle
    reference_data = pd.read_csv(ref_path)
    current_data = pd.read_csv(curr_path)

    if len(reference_data) > 5000:
        print("--- referans veri çok büyük, 5000 satırlık örneklem alınıyor... ---")
        reference_data = reference_data.sample(n=5000, random_state=42)

    # Ortak kolonları belirle (Drift analizi için referans ve current aynı kolonlara sahip olmalı)
    common_cols = [c for c in reference_data.columns if c in current_data.columns]
    
    # Raporu oluştur (Güncel sitedeki DataDriftPreset kullanımı)
    report = Report([
        DataDriftPreset() 
    ])

    # Analizi çalıştır
    my_eval = report.run(
        current_data=current_data[common_cols],
        reference_data=reference_data[common_cols] 
        
    )
    
    # Raporu HTML olarak kaydet
    my_eval.save_html(report_path)
    print(f"Rapor başarıyla oluşturuldu agası: {report_path}")

if __name__ == "__main__":
    run_phase4_monitoring()