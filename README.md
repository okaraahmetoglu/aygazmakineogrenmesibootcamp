# aygazmakineogrenmesibootcamp
aygazmakineogrenmesibootcamp
Kredi Kartları Skor Verisi İle Makine Öğrenmesi

Proje Url:
https://www.kaggle.com/code/osmankaraahmetolu/kredi-skor-verisi-makine-renmesi-bootcamp

Kaggle sitesindeki https://www.kaggle.com/datasets/ayushsharma0812/dataset-for-credit-score-classification linkindeki Kresi Skor verisetinin (Dataset for Credit Score Classification) keşifsel veri analizi aşağıda sunulmuştur.

Öncelikle panda kütüphanesi kullanılarak csv formatındaki veri kümesi  aşağıdaki kod scripti ile okunur.

import pandas as pd
df = pd.read_csv('/kaggle/input/kredi-skor-verisi/credit_score.csv')

df.info komutu veri kümesi ile ilgili genel bilgiler verir. Veri kümesinde 28 değişken vardır. Bu veri kümesinde 100000 kayıt vardır. Veri kümesinden Credit_Score değişkeni y olarak seçilir. Diğer değişkenlerden x seçilerek model olusturulur. Kolonların içerdiği değere göre alınan non-null countlarda farklı değerler olması verilerin null değerler içerdiğini göstermektedir.

#   Column                    Non-Null Count   Dtype  
---  ------                    --------------   -----  
 0   ID                        100000 non-null  object 
 1   Customer_ID               100000 non-null  object 
 2   Month                     100000 non-null  object 
 3   Name                      90015 non-null   object 
 4   Age                       100000 non-null  object 
 5   SSN                       100000 non-null  object 
 6   Occupation                100000 non-null  object 
 7   Annual_Income             100000 non-null  object 
 8   Monthly_Inhand_Salary     84998 non-null   float64
 9   Num_Bank_Accounts         100000 non-null  int64  
 10  Num_Credit_Card           100000 non-null  int64  
 11  Interest_Rate             100000 non-null  int64  
 12  Num_of_Loan               100000 non-null  object 
 13  Type_of_Loan              88592 non-null   object 
 14  Delay_from_due_date       100000 non-null  int64  
 15  Num_of_Delayed_Payment    92998 non-null   object 
 16  Changed_Credit_Limit      100000 non-null  object 
 17  Num_Credit_Inquiries      98035 non-null   float64
 18  Credit_Mix                100000 non-null  object 
 19  Outstanding_Debt          100000 non-null  object 
 20  Credit_Utilization_Ratio  100000 non-null  float64
 21  Credit_History_Age        90970 non-null   object 
 22  Payment_of_Min_Amount     100000 non-null  object 
 23  Total_EMI_per_month       100000 non-null  float64
 24  Amount_invested_monthly   95521 non-null   object 
 25  Payment_Behaviour         100000 non-null  object 
 26  Monthly_Balance           98800 non-null   object 
 27  Credit_Score              100000 non-null  object

df.describe() metodu ile veri alanları ile ilgili özet bilgilerle veri açıklanmaya ve analiz için fayda sağlayacak bilgiler edinilmeye çalışılacaktır. Bu komut ile veri kümesi ile ilgili kolon değerlerine kayıt sayısı, ortalama, frekans ve standarde deviation gibi hesaplamalar yapılarak veri analiz edilir.

Bu analiz sonucunda credit_score sahasının 3 farklı değer içermesi, sınıflandırmada tahmin edilecek y değişkeni olacağını göstermektedir. Analizden veri setinin Occupation ve Credit-Mix  isimli kategorik değişkenler içerdiği görülmektedir. 

df["Credit_Score"].describe()
count       100000
unique           3
top       Standard
freq         53174
Name: Credit_Score, dtype: object

df.head() komutu ile verilerin ilk 5 tanesi listelenir. Listelenen verileri inceleyerek verilerle ilgili gerekl, ön işleme süreeci belirlenecektir.

Veri setinde Nan değerler olması veri seti üzerinde nan değerleri düzeltecek bir veri temizleme sürecinin gerekli olduğunu göstermektedir.

Veri setinde Nan değerler olması veri seti üzerinde nan değerleri düzeltecek bir veri temizleme sürecinin gerekli olduğunu göstermektedir.

print(df.isnull().sum().sort_values(ascending=False))

Monthly_Inhand_Salary, Type_of_Loan, Name, Credit_History_Age, Num_of_Delayed_Payment, Amount_invested_monthly Num_Credit_Inquiries, Monthly_Balance sahalarında boş veriler olduğu yukarıda gözükmektedir.

Veri kümesinin görsel analizi için Profile-Report ile veri analiz edilir.

from ydata_profiling import ProfileReport
profile = ProfileReport(df)
profile.to_notebook_iframe()

Boş Kayıtlar
​
Profile report üzerinden verisetinde değeri null olan kolonlar görülmektedir.  Monthly_Inhand_Salary, Type_of_Loan, Name, Credit_History_Age, Num_of_Delayed_Payment,Amount_invested_monthly,Num_Credit_Inquiries,Monthly_Balance sahalarında null değerler olduğu profile report ve yukarıdaki null değerler sorgulamada gözükmektedir.
​
Bu durumda boş değerli kayıtlar silinerek veya istatistiki bir yöntemle doldurularak işletilen bir veri temizleme yönteminden sonra yola devam edilir.

Değişkenler
Veri setindeki 28 değişkenin, 13'ün metin, 6'sının kategorik, 8'inin nümerik olduğu profil raporundan görükmektedir. Bu analiz modele katılacak değişkenleri belirlemek açısından önemlidir. Nümerik ve kategorik değişkenler üzerine odaklanmak sağlıklı görülmektedir.

Korelasyon
Profile reportta verisetinde birbirleri arasında yüksek korelasyon olan kayıtlar olduğu görülmektedir. 
Delay_from_due_date ile Interest_Rate arasında ve Num_bank_accounts ile Delay_from_due_date arasında yüksek korelasyon olduğu görülmektedir. Yüksek korelasyonlu değişkenleri belirlemek model açısından değişken azaltmada önemlidir. Yüksek korelasyon olan değişkenlerin birlikte modelde olması, biri varken diğerinin modele katkısını azaltmakta, hatta karmaşıklığı artırdığı için sorun teşkil etmektedir. 

Kategorik Değişkenler

Aşağıdaki df.nunique() metodu çıktısında 6 kategorik değişken olduğu görülmektedir. Bu değişkenler, Month, Occupation, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour, Credit_Score dur.

Veri Sahalarındaki Distinct Kayıtlar.
Aşağıdaki df.nunique() metodu çıktısında 6 kategorik değişken olduğu görülmektedir. Bu değişkenler, Month, Occupation, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour, Credit_Score dur.

ID                          100000
Customer_ID                  12500
Month                            8
Name                         10139
Age                           1788
SSN                          12501
Occupation                      16
Annual_Income                18940
Monthly_Inhand_Salary        13235
Num_Bank_Accounts              943
Num_Credit_Card               1179
Interest_Rate                 1750
Num_of_Loan                    434
Type_of_Loan                  6260
Delay_from_due_date             73
Num_of_Delayed_Payment         749
Changed_Credit_Limit          4384
Num_Credit_Inquiries          1223
Credit_Mix                       4
Outstanding_Debt             13178
Credit_Utilization_Ratio    100000
Credit_History_Age             404
Payment_of_Min_Amount            3
Total_EMI_per_month          14950
Amount_invested_monthly      91049
Payment_Behaviour                7
Monthly_Balance              98792
Credit_Score                     3

Univariate Analysis

Annual_Income ile Credit_Score'un 


dfModel = df.copy()
komutu ile df değişkenindeki Dataframe'in dfModel değişkenine ataması yapılır.

Kurulması planlanan modelde anlamı olmayacak sahalar model dataframe'inden aşağıdaki komutla silinir.
dfModel = dfModel.drop(['Month','ID', 'Customer_ID',  'Name',  'SSN','Type_of_Loan','Credit_History_Age'], axis=1)

Veri Temizleme

Sahalarda anlam ifade etmeyen veriler bulunan veri satırları silinerek modelden atılır.

Payment_Behaviour değişkeni değerindeki !@9#%8 anlamsız değeri bulunan satır aşağıdaki komutla veri setinden silinmiştir.
dfModel = dfModel.loc[df['Payment_Behaviour'] != '!@9#%8'] 

!@9#%8 verisini içeren satırın silindiği yukarıda görülmektedir.
Bu işlem sonucunda veri sayısının 92400'e düştüğü aşağıda görülmektedir.

Boş Kayıtların Silinmesi

dropna() komutu ile veri setindeki nan değer içeren sahaların bulunduğu satırlar veri setinden silinir.

drop_dataframe = dfModel.dropna()

Aşağıda çalıştırılan drop_dataframe.isna() komutu verisetinde nan değer kalmadığını göstermektedir.
Bu işlem sonucunda veri kümesindeki kayıt sayısı 67676' ya düşmüştür.

drop_dataframe.isna()

Veri setinde Occupation ve Credit_Mix kategoriK değişkenlerinde aşağıda belirtilen anlamsız değerler belirlenmiştir. Occupation sahası ___ değeri olan satırları modelden silelim. Credit_Mix sahası ___ değeri olan satırları modelden silelim. Bu anlamsız değerleri içeren satırlar silindekten sonra kategorik değişkenleri etiketlemeye geçeçeceğiz.

Anlamsız değerler aşağıdaki komut seti temizlenir.

drop_dataframe = drop_dataframe.loc[df['Occupation'] != '_______'] 
drop_dataframe["Occupation"].unique()

drop_dataframe = drop_dataframe.loc[df['Credit_Mix'] != '_'] 
drop_dataframe["Credit_Mix"].unique()

Anlamsız değerler içeren satırlar veri kümesinden silindikten sonra veri kümesinde 50144  kayıt kalır.

Model verileri kullanılarak öğrenme admına geçmeden önceki son veri temizleme adımı modeldeki kategorik değişkenleri label encodin algoritması ile nümerik değerler ile etiketlemek olacaktır.

Modeldeki kategorik değişkenlerimiz, Occupation, Credit_Mix,Payment_of_Min_Amount,Payment_Behaviour,Credit_Score
Label Encoding yöntemi ile  0 dan başlayan sayı değerleri ile kodlanmıştır.

Etiketlenmiş Kategorik değişkenler kendi kolon isimleri ile kaydedilip, yeni kolonlar aşağıdaki komut ile silinir.
drop_dataframe.drop(['Occupation','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour','Credit_Score'],inplace=True, axis=1)

drop_dataframe.rename(columns={'COccupation': 'Occupation', 'CCredit_Mix': 'Credit_Mix', 'CPayment_of_Min_Amount': 'Payment_of_Min_Amount','CPayment_Behaviour': 'Payment_Behaviour','CCredit_Score': 'Credit_Score'}, inplace=True)

Oluşan son veri kümesinde nümerik olmayan değer içeren satırlar silinir.

numeric_df = drop_dataframe.apply(pd.to_numeric, errors='coerce')
numeric_df =numeric_df.dropna()
numeric_df.info()

Veri kümesi kolonlarında nümerik olmayan değer içeren satırlar yukarıdaki komut ile temizlendikten sonra elimizde 37846 kayıt kalır.


Aşağıdaki kod parçası ile m modelin x ve y veri kümesi oluşturulur.

y_DataSet = numeric_df["Credit_Score"]
x_DataSet = numeric_df.copy()
x_DataSet = x_DataSet.drop(["Credit_Score"], axis=1)

Model 19 değişkenden oluşur. Veri kümesindeki 19 değişken kullanılarak oluşturulan model Credit_Score y değişkenini tahmin etmeye çalışır.

Gözetimli Öğrenme Algoritmaları:

Decision Tree Algoritması
