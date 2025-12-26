# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="1018" height="570" alt="530366673-5132088b-cec1-438a-825e-002554eb6d27" src="https://github.com/user-attachments/assets/33c30941-9015-406c-8a96-c23a659827ff" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="740" height="352" alt="530366852-b901dfc1-4426-4790-ac2c-3fd7c323dfd5" src="https://github.com/user-attachments/assets/06444960-8072-4a4a-b124-d3b4c0dd33b2" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="722" height="528" alt="530367068-47b385d9-1e07-4220-b666-6b1a10420c93" src="https://github.com/user-attachments/assets/c2fdf432-e3ae-46e0-8850-0edb21c58df5" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="644" height="569" alt="530367135-99847ba6-e0ec-4124-a489-19e76378db44" src="https://github.com/user-attachments/assets/0216ff0d-a904-48b2-a81a-4c20f62bab5d" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="550" height="441" alt="530367220-7099be56-6432-4764-8f5e-2a257574f686" src="https://github.com/user-attachments/assets/4df6aa9f-f0aa-4a22-975c-4ce2633aca11" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="828" height="456" alt="530367337-a48613dd-3bd5-46d8-9ba3-a5a828f22a4c" src="https://github.com/user-attachments/assets/505e3baa-4843-499b-951b-1914fd23a0af" />

```
pip install --upgrade category_encoders
```

<img width="1382" height="428" alt="530367443-13c46a87-ff1b-468b-8584-ca48d890f7f5" src="https://github.com/user-attachments/assets/3296570b-86cb-41b3-8962-d19ac6d3cf86" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

<img width="622" height="439" alt="530367591-b0f2ba98-e977-4ddf-94df-63acf23abe29" src="https://github.com/user-attachments/assets/4c4e1bc7-f81f-46ca-a988-4182af047b6a" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="620" height="444" alt="530367726-fc039f09-e06c-49f7-80e9-a4c2f47e7624" src="https://github.com/user-attachments/assets/136b4283-3bfd-4c86-bf7f-31aa622c4fc7" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="877" height="441" alt="530367811-4e6dea0c-5ec9-446a-a9c4-ed1ff026bf4a" src="https://github.com/user-attachments/assets/0448fc48-00fa-4c5f-b863-51b6fad26224" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="709" height="445" alt="530367915-4327a2e2-782b-4eda-ae2b-9d81b3c1149c" src="https://github.com/user-attachments/assets/142c5420-c418-4163-a43d-b293677e1dc8" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

<img width="986" height="498" alt="530367998-f0a4f625-972a-463f-afad-82c4f9dc6d8f" src="https://github.com/user-attachments/assets/7fb4f099-5e56-43b7-b07c-cf1bc8fe00b8" />

```
df.skew()
```

<img width="434" height="243" alt="530368088-4added85-e16f-475b-b7e2-00bb7cb46bce" src="https://github.com/user-attachments/assets/ff47629f-d161-4af4-921f-fad01c68b920" />

```
np.log(df["Highly Positive Skew"])

```

<img width="470" height="557" alt="530368142-9ec25108-e1ab-4c8c-8985-1fc93e6c3f2b" src="https://github.com/user-attachments/assets/32eba1fb-7bf2-49e9-88ac-fc2b298aa878" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="536" height="586" alt="530368176-8aca222e-ad97-4446-83b7-dfe0a4b6211e" src="https://github.com/user-attachments/assets/3bffb5b1-1649-4931-a539-9bd82bee7a33" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="597" height="577" alt="530368218-fa4dd03e-a21f-4895-987a-8548d6bf47f6" src="https://github.com/user-attachments/assets/c72b4f14-ae8b-4970-8019-328ce033f5f7" />

```
np.square(df["Highly Positive Skew"])
```

<img width="651" height="567" alt="530368251-9538676e-9f7c-4aa5-956f-2ea59bf4d989" src="https://github.com/user-attachments/assets/daabac87-aa88-4ddf-a6e4-c496374d63ce" />

```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```

<img width="1276" height="517" alt="530368301-9a195b1f-3ea7-40db-82e4-c17a83209587" src="https://github.com/user-attachments/assets/0ac4ee30-55dc-4912-9b7f-44e0e419a5a3" />

```
df.skew()
```

<img width="515" height="294" alt="530368369-98a723ed-6d2c-4411-97fe-6c8174d937c9" src="https://github.com/user-attachments/assets/5a281787-a8af-4d5c-98ce-66917303c136" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="615" height="354" alt="530368424-390687e3-8fa5-4202-807d-1275c0bf00a0" src="https://github.com/user-attachments/assets/cb9d4863-0aed-4c49-b08f-602300729000" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1343" height="556" alt="530368469-c0748f2e-1811-4561-bd0e-20ca3a41cef6" src="https://github.com/user-attachments/assets/b7c1c017-ab88-401b-a444-4bdfd3191e43" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="910" height="561" alt="530368534-aeda7780-d619-4ba8-b47c-348dbc12ca38" src="https://github.com/user-attachments/assets/441e03a1-cd98-4745-9949-b7ba95cc6c05" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="929" height="562" alt="530368583-a7e48561-85d2-49cf-88fc-3d9c31514f2f" src="https://github.com/user-attachments/assets/59b2cca3-5f28-473d-ac53-da3eca775c1d" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="1062" height="557" alt="530368645-6d003170-2131-4b87-bacf-2eaf2b7e0ab1" src="https://github.com/user-attachments/assets/0cd17437-9f59-4e85-a375-107690dac093" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="828" height="546" alt="530368743-250a9ed1-d7b9-4688-9b98-a522a70b3ec2" src="https://github.com/user-attachments/assets/95151919-97ff-4a23-851e-1b49685f7ad0" />

```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```

<img width="1360" height="620" alt="530368820-05348b27-4eb2-4a1a-9905-c6d67365c70d" src="https://github.com/user-attachments/assets/896d3fea-1168-4407-8656-da9e40e9ec40" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

<img width="1090" height="558" alt="530368883-f835196c-0332-4668-89fe-106d691ad547" src="https://github.com/user-attachments/assets/67d61120-ea6e-4b5f-ac9a-a18afa2ac8fe" />


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

<img width="831" height="559" alt="530368970-7fa1f5a1-900b-4fbd-a164-7972c422522c" src="https://github.com/user-attachments/assets/9cc691d3-632f-42f2-b299-c44d2dd807f2" />


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
