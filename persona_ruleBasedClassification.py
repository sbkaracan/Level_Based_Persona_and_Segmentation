import pandas as pd
df = pd.read_csv('datasets/persona.csv')

df.head()
""" OUTPUT:
   PRICE   SOURCE   SEX COUNTRY  AGE
0     39  android  male     bra   17
1     39  android  male     bra   17
2     49  android  male     bra   17
3     29  android  male     tur   17
4     49  android  male     tur   17
"""

df.shape
# OUTPUT: (5000, 5)

df.info()
""" OUTPUT:
<class 'pandas.core.frame.DataFrame'>
Int64Index: 5000 entries, 0 to 4999
Data columns (total 5 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   PRICE    5000 non-null   int64 
 1   SOURCE   5000 non-null   object
 2   SEX      5000 non-null   object
 3   COUNTRY  5000 non-null   object
 4   AGE      5000 non-null   int64 
dtypes: int64(2), object(3)
memory usage: 234.4+ KB
"""

# Observation of total earned money at the breakdown of country, source, sex, age.
df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "sum"})

""" OUTPUT:
                            PRICE
COUNTRY SOURCE  SEX    AGE       
bra     android female 15    1355
                       16    1294
                       17     642
                       18    1387
                       19    1021
                           ...
usa     ios     male   42     242
                       50     156
                       53      68
                       55      29
                       59     186
[348 rows x 1 columns]
"""

# Previous observation is sorted from largest to smallest and assigned to agg_df variable.
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "sum"}).sort_values("PRICE", ascending=False)
agg_df.head()
""" OUTPUT:

                             PRICE
 COUNTRY SOURCE  SEX    AGE
 usa     android male   15    3917
 bra     android male   19    2606
 usa     ios     male   15    2496
         android female 20    2190
 deu     ios     female 16    2169
"""


# Since we've used groupby; our all features used in groupby, became indexes. To make them columns again, we use reset_index() function.
agg_df = agg_df.reset_index()
agg_df.head()
""" OUTPUT:
   COUNTRY   SOURCE     SEX  AGE  PRICE
 0     usa  android    male   15   3917
 1     bra  android    male   19   2606
 2     usa      ios    male   15   2496
 3     usa  android  female   20   2190
 4     deu      ios  female   16   2169
"""

# Since we separate our customers into segments, we need to convert age feature to a categorical feature.
# We will determine some intervals to make assignments. For example our intervals might be: [0, 18] [19, 23] [24, 30] [31, 40] [41, 70]

bins = [0, 19, 24, 31, 41, agg_df["AGE"].max()] # interval points are determined here

mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())] # intervals are labeled.

agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels) # intervals are made with the help of cut function.
agg_df.head()
"""
  COUNTRY   SOURCE     SEX  AGE  PRICE age_cat
0     usa  android    male   15   3917    0_18
1     bra  android    male   19   2606    0_18
2     usa      ios    male   15   2496    0_18
3     usa  android  female   20   2190   19_23
4     deu      ios  female   16   2169    0_18
"""


# After adjusting the age intervals, now we will create level based categories.
# We will create a new column on dataframe and we'll have the segments of customers here. For example : USA_ANDROID_MALE_0_18
# List comprehension will be used to create names of the segments.

# under the customers_level_based column we combine other columns with "_". (USA_ANDROID_MALE_0_18)
agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df.head()
"""
  COUNTRY   SOURCE     SEX  AGE  PRICE age_cat     customers_level_based
0     usa  android    male   15   3917    0_18     USA_ANDROID_MALE_0_18
1     bra  android    male   19   2606    0_18     BRA_ANDROID_MALE_0_18
2     usa      ios    male   15   2496    0_18         USA_IOS_MALE_0_18
3     usa  android  female   20   2190   19_23  USA_ANDROID_FEMALE_19_23
4     deu      ios  female   16   2169    0_18       DEU_IOS_FEMALE_0_18
"""

# Having only important columns. Others are not needed anymore.
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()
"""
      customers_level_based  PRICE
0     USA_ANDROID_MALE_0_18   3917
1     BRA_ANDROID_MALE_0_18   2606
2         USA_IOS_MALE_0_18   2496
3  USA_ANDROID_FEMALE_19_23   2190
4       DEU_IOS_FEMALE_0_18   2169
"""


# There will be customers at the same level. For example there could be more than 1 USA_ANDROID_MALE_0_18 in this dataframe.
# To make it singularized, we will use groupby function on customers_level_based column.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

# reseting index
agg_df = agg_df.reset_index()
agg_df.head()

"""
      customers_level_based        PRICE
0   BRA_ANDROID_FEMALE_0_18  1139.800000
1  BRA_ANDROID_FEMALE_19_23  1070.600000
2  BRA_ANDROID_FEMALE_24_30   508.142857
3  BRA_ANDROID_FEMALE_31_40   233.166667
4  BRA_ANDROID_FEMALE_41_66   236.666667
"""

# To check that if there is only 1 level for each segment. This is what we need.
agg_df["customers_level_based"].value_counts()
""" OUTPUT:
BRA_IOS_FEMALE_19_23        1
FRA_IOS_FEMALE_31_40        1
TUR_IOS_MALE_0_18           1
BRA_ANDROID_FEMALE_19_23    1
CAN_IOS_MALE_41_66          1
                           ..
DEU_ANDROID_MALE_24_30      1
USA_ANDROID_FEMALE_41_66    1
CAN_ANDROID_FEMALE_24_30    1
BRA_ANDROID_FEMALE_41_66    1
FRA_IOS_MALE_31_40          1
Name: customers_level_based, Length: 108, dtype: int64
"""

agg_df.head()
""" OUTPUT:
      customers_level_based        PRICE
0   BRA_ANDROID_FEMALE_0_18  1139.800000
1  BRA_ANDROID_FEMALE_19_23  1070.600000
2  BRA_ANDROID_FEMALE_24_30   508.142857
3  BRA_ANDROID_FEMALE_31_40   233.166667
4  BRA_ANDROID_FEMALE_41_66   236.666667
"""


# Dividing customers into segments. It will be divided by price. Qcut will be used to divide. Highest prices will be segment A and lowest prices will be D.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
""" OUTPUT:
       customers_level_based        PRICE SEGMENT
0    BRA_ANDROID_FEMALE_0_18  1139.800000       A
1   BRA_ANDROID_FEMALE_19_23  1070.600000       A
2   BRA_ANDROID_FEMALE_24_30   508.142857       A
3   BRA_ANDROID_FEMALE_31_40   233.166667       C
4   BRA_ANDROID_FEMALE_41_66   236.666667       C
5      BRA_ANDROID_MALE_0_18  1543.400000       A
6     BRA_ANDROID_MALE_19_23   569.400000       A
7     BRA_ANDROID_MALE_24_30   551.333333       A
8     BRA_ANDROID_MALE_31_40   454.250000       B
9     BRA_ANDROID_MALE_41_66   155.666667       D
10       BRA_IOS_FEMALE_0_18   647.600000       A
11      BRA_IOS_FEMALE_19_23   444.250000       B
12      BRA_IOS_FEMALE_24_30   237.666667       C
13      BRA_IOS_FEMALE_31_40   405.750000       B
14      BRA_IOS_FEMALE_41_66   240.750000       C
15         BRA_IOS_MALE_0_18   931.200000       A
16        BRA_IOS_MALE_19_23   680.000000       A
17        BRA_IOS_MALE_24_30    58.000000       D
18        BRA_IOS_MALE_31_40   204.000000       C
19        BRA_IOS_MALE_41_66   199.800000       C
20   CAN_ANDROID_FEMALE_0_18   428.750000       B
21  CAN_ANDROID_FEMALE_19_23   126.000000       D
22  CAN_ANDROID_FEMALE_24_30    19.000000       D
23  CAN_ANDROID_FEMALE_41_66   680.000000       A
24     CAN_ANDROID_MALE_0_18   360.000000       B
25    CAN_ANDROID_MALE_19_23   374.000000       B
26    CAN_ANDROID_MALE_24_30   322.000000       B
27    CAN_ANDROID_MALE_41_66   263.000000       C
28       CAN_IOS_FEMALE_0_18   315.500000       C
29      CAN_IOS_FEMALE_24_30   349.000000       B
"""

# Average of each segment
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})
"""
              PRICE
SEGMENT            
D        121.261728
C        249.730864
B        387.284832
A        886.461464
"""


# Analyzing segment C
agg_df[agg_df["SEGMENT"] == "C"]
"""
        customers_level_based       PRICE SEGMENT
3    BRA_ANDROID_FEMALE_31_40  233.166667       C
4    BRA_ANDROID_FEMALE_41_66  236.666667       C
12       BRA_IOS_FEMALE_24_30  237.666667       C
14       BRA_IOS_FEMALE_41_66  240.750000       C
18         BRA_IOS_MALE_31_40  204.000000       C
19         BRA_IOS_MALE_41_66  199.800000       C
27     CAN_ANDROID_MALE_41_66  263.000000       C
28        CAN_IOS_FEMALE_0_18  315.500000       C
35   DEU_ANDROID_FEMALE_19_23  240.500000       C
53   FRA_ANDROID_FEMALE_19_23  217.000000       C
54   FRA_ANDROID_FEMALE_24_30  223.000000       C
56      FRA_ANDROID_MALE_0_18  281.666667       C
62       FRA_IOS_FEMALE_24_30  232.333333       C
64          FRA_IOS_MALE_0_18  235.250000       C
65         FRA_IOS_MALE_19_23  287.000000       C
67         FRA_IOS_MALE_31_40  203.000000       C
72   TUR_ANDROID_FEMALE_31_40  215.000000       C
74      TUR_ANDROID_MALE_0_18  247.000000       C
75     TUR_ANDROID_MALE_19_23  304.666667       C
79        TUR_IOS_FEMALE_0_18  254.500000       C
82       TUR_IOS_FEMALE_31_40  231.000000       C
87         TUR_IOS_MALE_41_66  305.000000       C
92   USA_ANDROID_FEMALE_41_66  192.800000       C
96     USA_ANDROID_MALE_31_40  255.666667       C
97     USA_ANDROID_MALE_41_66  306.500000       C
102      USA_IOS_FEMALE_41_66  267.800000       C
106        USA_IOS_MALE_31_40  312.500000       C
"""


########################## Questions: ###############################33

# What would we expect from a Turkish, 33 years old, Android user female?

new_user = "TUR_ANDROID_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] == new_user]
"""
       customers_level_based  PRICE SEGMENT
72  TUR_ANDROID_FEMALE_31_40  215.0       C
"""


# What would we expect from a French, 35 years old, IOS user male?
new_user = "FRA_IOS_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] == new_user]
"""
   customers_level_based  PRICE SEGMENT
63  FRA_IOS_FEMALE_31_40  165.0       D
"""