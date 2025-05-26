import pandas as pd


def csv_to_gobnilp(name):

    with open("data/%s.csv" % (name), 'r') as f:
        header = f.readline().strip().replace(',', ' ')

    with open("data/convert_table/%s.csv" % (name), 'r') as f:
        f.readline()
        data = f.read().replace(',', ' ')

    df = pd.read_csv("data/convert_table/%s.csv" % (name))
    print(df.columns)
    values = [len(df[col].unique()) for col in df.columns]
    print(values)
    s = str(values)[1:-1].replace(',', '')
    print(s)

    with open("structure_learning/gobnilp/data/%s.dat" % (name), 'w') as f:
        f.write(header)
        f.write('\n')
        f.write(s)
        f.write('\n')
        f.write(data)

if __name__ == '__main__':
    csv_to_gobnilp("cancer_5000")
    csv_to_gobnilp("earthquake_5000")
    csv_to_gobnilp("asia_5000")
    csv_to_gobnilp("alarm_5000")
    csv_to_gobnilp("insurance_5000")
    csv_to_gobnilp("water_5000")
    csv_to_gobnilp("cancer_5000_0.01")
    csv_to_gobnilp("cancer_5000_0.05")
    csv_to_gobnilp("alarm_5000_0.01")
    csv_to_gobnilp("alarm_5000_0.05")