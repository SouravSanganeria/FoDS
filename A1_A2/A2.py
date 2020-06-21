from DATA import UnlimitedDataWorks
from MODELS import RegressionModel
import pandas as pd


columns = ["junk", "lat", "lon", "alt"]
raw_df = pd.read_csv("3D_spatial_network.txt", sep=',', header=None,
                     names=columns).drop("junk", 1).sample(frac=1)

deg = input("Enter the Degree of the Polynomial:")
pre_processor = UnlimitedDataWorks(deg=int(deg))
X_train, Y_train, x_val, y_val, x_test, y_test = pre_processor.train_test_split(raw_df, normalize=True)

model = RegressionModel(N=pre_processor.count,
                        X=X_train,
                        Y=Y_train,
                        x=x_test,
                        y=y_test,
                        xval=x_val,
                        yval=y_val)

model.fit()
# model.gradient_descent(lr=3e-6)
# model.stocastic_gradient_descent(50000, lr=0.1)
# model.gradient_descent_L1_reg(lr=3e-6)
# model.gradient_descent_L2_reg(lr=3e-6)
