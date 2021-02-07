import seaborn as sns
import matplotlib.pyplot as plt
from trees import calc_score

class Model():
    def __init__(self, train_df, test_df, label = ""):
        self.train_df = train_df
        self.test_df  = test_df
        self.label = label

    def fit(self):
        raise "Fit not implemented"
        pass
    
    def predict_df(self, df):
        raise "predict not implemented"

    def compute_errors(self, df):
        modelY = self.predict_df(df)
        target = df["target"].to_numpy().reshape(-1, 1)
        return modelY-target

    def compute_score(self, df):
        ymodel = self.predict_df(df)
        target = df["target"].to_numpy().reshape(-1, 1)
        return calc_score(target, ymodel, target.mean())[0]

    def error_histogram(self, df):
        err = self.compute_errors(df)
        sns.displot(err)
        plt.show()
        # if both:
        #     train_err = self.compute_errors(self.train_df)
        #     test_err = self.compute_errors(self.test_df)
        #     train_data = np.full_like(train_err, "train", dtype=str)
        #     errTrain = pd.DataFrame(np.column_stack((train_data,train_err)), columns=['dataset', 'err'])
        #     #err = pd.DataFrame(np.column_stack((err_train,err_test)), columns=['train', 'test'])
        #     sns.displot(errTrain, x = "err")
        #     plt.show()
        # else:
        #     err = self.compute_errors(df)
        #     sns.displot(err)
        #     plt.show()