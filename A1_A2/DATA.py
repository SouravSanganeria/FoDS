import pandas as pd


class UnlimitedDataWorks:

    def __init__(self, deg):
        self.count = 0
        self.exp = []
        for i in range(deg+1):
            for j in range(deg+1):
                if i+j <= deg:
                    self.exp.append((i, j))

    def train_test_split(self, df, normalize=False):
        data = pd.DataFrame([])
        print("Starting Reality Marble...")
        if normalize == True:
            for col in df.columns:
                mx = df[col].max()
                mn = df[col].min()
                df[col] = (df[col] - mn)/(mx - mn)
        for (a, b) in self.exp:
            res = ((df["lat"] ** b) * (df["lon"] ** a))
            data.insert(self.count, "feat " + str(a) + str(b), res, True)
            self.count += 1
        
        # generate a 70-20-10 split on the data:
        X = data[:304113]
        Y = df["alt"][:304113]
        xval = data[304113:391088]
        yval = df["alt"][304113:391088]
        x = data[391088:]
        y = df["alt"][391088:]
        return (X, Y, xval, yval, x, y)
