# WDM-Ex.No-5-CUSTOMER-SEGMENTATION-ANALYSIS-USING-CLUSTERING-
## Name: Kavipriya SP
## Date:
**Aim:** 
To perform an unsupervised clustering of data on the customer's records from a groceries firm's database.  
**Description:** 
Customer segmentation is the practice of separating customers into groups that reflect similarities among customers in 
each cluster. I will divide customers into segments to optimize the significance of each customer to the business. To 
modify products according to distinct needs and behaviors of the customers. It also helps the business to cater to the 
concerns of different types of customers. <br>
**Program:** 
**Loading the dataset**
```
data = pd.read_csv("/content/marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))
data.head()
```
**Information on features**  
```
data.info()
```
**To remove the NA values** 
```
data = data.dropna() 
print("The total number of data-points after removing the rows with missing values are:", len(data)) 
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"]) 
dates = [] 
for i in data["Dt_Customer"]: 
    i = i.date() 
    dates.append(i)
```  
**Dates of the newest and oldest recorded customer** 
```
print("The newest customer's enrolment date in therecords:",max(dates)) 
print("The oldest customer's enrolment date in the records:",min(dates))
``` 
**Created a feature "Customer_For"** 
```
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")
print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())
```
## Performing the following steps to engineer some new features: 
 Extract the "Age" of a customer by the "Year_Birth" indicating the birth year of the respective person. 
 Create another feature "Spent" indicating the total amount spent by the customer in various categories over the 
span of two years. 
 Create another feature "Living_With" out of "Marital_Status" to extract the living situation of couples. 
 Create a feature "Children" to indicate total children in a household that is, kids and teenagers. 
 To get further clarity of household, creating feature indicating "Family_Size" 
 Create a feature "Is_Parent" to indicate parenthood status. 
 Lastly, create three categories in the "Education" by simplifying its value counts. 
 Dropping some of the redundant features 
## Feature Engineering 
```
# Age of customer
data["Age"] = 2021 - data["Year_Birth"]

# Total amount spent
data["Spent"] = data[
    ["MntWines", "MntFruits", "MntMeatProducts",
     "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
].sum(axis=1)

# Living situation
data["Living_With"] = data["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Absurd": "Alone", "Widow": "Alone",
    "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"
})

# Children and family size
data["Children"] = data["Kidhome"] + data["Teenhome"]
data["Family_Size"] = data["Living_With"].map(
    {"Alone": 1, "Partner": 2}
) + data["Children"]

data["Is_Parent"] = np.where(data["Children"] > 0, 1, 0)

# Simplify education categories
data["Education"] = data["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate", "PhD": "Postgraduate"
})

# Rename columns
data = data.rename(columns={
    "MntWines": "Wines", "MntFruits": "Fruits",
    "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"
})

# Remove unnecessary columns
data = data.drop(columns=[
    "Marital_Status", "Dt_Customer", "Z_CostContact",
    "Z_Revenue", "Year_Birth", "ID"
])
```
## DATA PREPROCESSING 
The following steps are applied to preprocess the data: 
 Label encoding the categorical features. 
 Scaling the features using the standard scaler 
```
object_cols = data.select_dtypes(include="object").columns

for col in object_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))
```
## creating a subset of dataframe by dropping the features on deals accepted and promotions 
```
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 
'Response'] 
ds = ds.drop(cols_del, axis=1) 
#Scaling 
scaler = StandardScaler() 
scaler.fit(ds) 
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns ) 
print("All features are now scaled") 
#Scaled data to be used for reducing the dimensionality 
print("Dataframe to be used for further modelling:") 
scaled_ds.head()
``` 
## DIMENTIONALITY REDUCTION 
Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining 
a set of principal variables. 
Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing 
interpretability but at the same time minimizing information loss. 
## Steps in this section: 
 Dimensionality reduction with PCA 
 Plotting the reduced dataframe 
```
pca = PCA(n_components=3)
PCA_ds = pd.DataFrame(
    pca.fit_transform(scaled_ds),
    columns=["col1", "col2", "col3"],
    index=data.index
)

print("\nPCA completed.")
print("Explained variance:", pca.explained_variance_ratio_)

# Display 3D PCA projection
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"])
ax.set_title("3D Projection After PCA")
plt.show()
```
 ## CLUSTERING 
Agglomerative clustering is a hierarchical clustering method. It involves merging examples until the desired number 
of clusters is achieved. 
Steps involved in the Clustering: 
 Elbow Method to determine the number of clusters to be formed. 
 Clustering via Agglomerative Clustering 
 Examining the clusters formed via scatter plot 
## Quick examination of elbow method to find numbers of clusters to make. 
```print("\nElbow Method to determine the number of clusters:")

inertia = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(PCA_ds)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()
```
## EVALUATING MODELS 
```
# AGGLOMERATIVE CLUSTERING
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)

PCA_ds["Clusters"] = yhat_AC
data["Clusters"] = yhat_AC

# Plot clusters in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"],
    c=PCA_ds["Clusters"], cmap="viridis", marker="o"
)
ax.set_title("Customer Clusters")
plt.show()
```
## Income vs spending plot shows the clusters pattern 
 group 0: high spending & average income 
 group 1: high spending & high income 
 group 2: low spending & low income 
 group 3: high spending & low income 
```
sns.scatterplot(
    data=data, x="Spent", y="Income",
    hue="Clusters", palette=palette
)
plt.title("Cluster Profile: Income vs Spending")
plt.show()

# Spending distribution by cluster
sns.boxplot(
    data=data, x="Clusters", y="Spent", hue="Clusters",
    palette=palette, legend=False
)
plt.title("Spending Distribution by Cluster")
plt.show()

## FINAL RESULT
print("\nCustomers in each cluster:")
print(data["Clusters"].value_counts().sort_index())

print("\nRESULT: Customer segmentation completed successfully.")
```
 
## Output: 
<img width="542" height="748" alt="image" src="https://github.com/user-attachments/assets/79719f69-b902-4d40-b687-251d3a18a3a6" />
 
 <img width="720" height="736" alt="image" src="https://github.com/user-attachments/assets/b4d84833-4430-43a0-8d9a-6a442c1a979b" />

<img width="771" height="603" alt="image" src="https://github.com/user-attachments/assets/e5fb0b4c-11a3-40a9-add3-eee3a3e6879d" />

<img width="717" height="640" alt="image" src="https://github.com/user-attachments/assets/e6c92a2d-8bea-47e9-927f-7e19ac6444a4" />

<img width="872" height="622" alt="image" src="https://github.com/user-attachments/assets/43af2264-b226-4617-927d-2128072020d3" />

<img width="730" height="647" alt="image" src="https://github.com/user-attachments/assets/7b2803e3-e24f-4cc3-8ccf-dcdbd725d800" />

<img width="773" height="513" alt="image" src="https://github.com/user-attachments/assets/4f9b436a-d15a-44f3-886e-2402670dfb48" />

<img width="725" height="517" alt="image" src="https://github.com/user-attachments/assets/dc751501-bc51-4915-86f7-2f2d0cfcb63c" />

<img width="908" height="585" alt="image" src="https://github.com/user-attachments/assets/3d188bd3-bcb3-4778-8ce9-c519844dafec" />
 
## Result: 
Thus, dimensionality reduction and agglomerative clustering have been performed for profiling customers in clusters according to their family structures and income/spending. 
