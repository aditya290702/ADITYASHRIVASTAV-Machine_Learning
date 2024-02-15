#Write a program to partition a dataset (simulated data for regression)  into two parts,
# based on a feature (BP) and for a threshold, t = 80.
# Generate additional two partitioned datasets based on different threshold values of t = [78, 82].
import pandas as pd

def split():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    BP = df['BP']
    print("Enter a Threshold Value")
    a = int(input())
    Less_than_a = []
    More_than_a = []

    for ele in BP:
        if (ele > a):
            More_than_a.append(ele)
        elif (ele < a):
            Less_than_a.append(ele)
    print("more than 80")
    print(pd.DataFrame(More_than_a),end=" ")
    print("Less than 80")
    print(pd.DataFrame(Less_than_a))

def main():
    split()

if __name__ == '__main__':
    main()
