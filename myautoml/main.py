from api.automl import AutoML

def main():
    automl = AutoML()
    automl.run_automl()
    results = automl.get_results()
    print("AutoML process completed. Results:")
    print(results)

if __name__ == "__main__":
    main()