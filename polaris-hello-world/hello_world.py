import polaris as po


def main():
    benchmark = po.load_benchmark("polaris/hello-world-benchmark")
    train, test = benchmark.get_train_test_split()
    print(test[0])


if __name__ == "__main__":
    main()
