from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



def label_encoder(df,target):
    le=LabelEncoder()
    df_copy=df.copy()
    df_copy[target]=le.fit_transform(df_copy[target])
    return le, df_copy


def split_dataset (df,target):

    df_copy=df.copy()
    X=df_copy.drop(target,axis=1)
    y=df_copy[target]

    X_train,X_test,y_train,y_test= train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train,X_test,y_train,y_test

def baseline ():

    model= LogisticRegression(max_iter=200)
    model