def create_report(results, output_path):
    with open(output_path, 'w') as f:
        f.write("AutoML Report\n")
        f.write("====================\n")
        for model_name, metrics in results.items():
            f.write(f"Model: {model_name}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")

def save_report(report, filename):
    with open(filename, 'w') as f:
        f.write(report)