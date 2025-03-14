import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

# region World Happiness Report
df = pd.read_csv(filepath_or_buffer='data/world_happiness_report.csv')

df.rename(
    columns={
        'Country or region': 'Country', 
        'Overall rank' : "Rank",
        "GDP per capita": "GDP",
        "Social support": "Support",
        "Healthy life expectancy": "LifeExp",
        "Freedom to make life choices": "Freedom",
        "Generosity": "Generosity",
        "Perceptions of corruption": "Corruption"
    },
    inplace=True)

df.drop('Rank', axis=1, inplace=True)

# GDP değeri 1.4’ten büyük ve LifeExp değeri 1.0’dan küçük olan ülkeleri listeleme
# 1.yol
filtered_df = df[(df['GDP'] >= 1.4)  & (df['LifeExp'] <= 1.0)]
print(filtered_df)

# 2.yol
print(df.query('GDP > 1.4 & LifeExp < 1.0'))


# Freedom skoru 0.6'dan yüksek olan ülkeleri ve freedom değerini listeleme
print(df[df['Freedom'] > 0.6][['Country', 'Freedom']])


# Mutluluk skoru en yüksek 5 ülkeyi nasıl sıralayabilirsiniz?
max_score = df['Score'].sort_values(ascending=False).head(5)
print(max_score)


# Ülkeleri GDP ortalamsına göre 3 gruba (düşük, eşit, yüksek) ayırarak ortalama mutluluk skorlarını hesaplama
gdp_mean = df['GDP'].mean()

# def categorize_gdp(gdp):
#     if gdp < gdp_mean:
#         return 'Düşük'
#     elif gdp == gdp_mean:
#         return 'Eşit'
#     else:
#         return 'Yüksek'
    
# df['Categorize_GDP'] = df['GDP'].apply(categorize_gdp)

df['Categorize_GDP'] = df['GDP'].apply(lambda x: 'Düşük' if x < gdp_mean else 'Yüksek' if x > gdp_mean else 'Eşit')
print(df.groupby('Categorize_GDP')['Score'].mean())
   

# Ortalama freedom değerinin altında kalan ülkeleri gruplandırarak ortalama Generosity değerini hesaplama
low_freedom = df[df['Freedom'] < df['Freedom'].mean()]
print(low_freedom.groupby('Country')['Generosity'].mean())

# endregion

 
# region Superstore    turkcell deki verilerle birleştir

df = pd.read_csv(filepath_or_buffer='data/superstore_sales.csv')
df.columns = df.columns.str.replace(' ', '_')
print(df.columns)


# Row ID 5 ile 10 arasındaki satırların müşteri adı ve satış bilgilerini listeleme
print(df.loc[5:10, ['Customer_Name', 'Sales']])


# Satış sütununun temel istatistiklerini (ortalama, standart sapma, min, max, vb.) hesaplama
print(df['Sales'].describe())


# Her bir müşteri adı için toplam satış miktarını hesaplama ve en çok satış yapan 10 müşteriyi listeleme
top_customers = df.groupby('Customer_Name')['Sales'].sum().sort_values(ascending=False).head(10)
print(top_customers)


# Her bir kategori ve alt kategori için toplam satış ve ortalama satış işlemlerini hesaplama 
print(
    df.groupby(['Category', 'Sub-Category'])['Sales']
    .agg(['sum', 'mean'])
    )


# Bölge sütununu kullanarak, her bir bölgedeki min ve max satış fiyatını hesaplama
print(
    df.groupby('Region')['Sales']
      .agg(['min', 'max'])
      )  


# Sipariş Tarihi 2018'in ilk çeyreğinde olan ve "Kırtasiye" kategorisine ait olan siparişleri bulma
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d/%m/%Y')
print(
   df[(df['Order_Date'].between('01/01/2018' , '31/03/2018')) & (df['Category'] == 'Office Supplies')]
   [['Customer_Name','Product_Name', 'Order_Date', 'City', 'Sales']].head(10)
)


# Zaman serisi verilerini (Sipariş Tarihi ve Satış) kullanarak trendleri ve mevsimselliği gösteren grafiği oluşturma
time_series = df.groupby('Order_Date')['Sales'].sum()
decomposition = seasonal_decompose(time_series, model='additive', period=30)

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(decomposition.observed, label='Gözlemlenen')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Mevsimsellik')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Rastgele')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



# Farklı müşteri segmentlerinin satın alma davranışlarının aralarındaki korelasyonu tespit etme ve  görselleştirme

# 1.adım: Her bir müşteri segmenti için satın alınan ürünlerin pivot tablosunu oluşturulur.
# 2.adım: Bu pivot tabloyu kullanarak ürünler arasındaki korelasyon matrisi hesaplanır ve görselleştirilir

segment_product_sales = df.pivot_table(index='Customer_ID', columns='Product_Name', values='Sales', aggfunc='sum', fill_value=0)
correlation_matrix = segment_product_sales.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Ürünler Arasındaki Korelasyon Matrisi')
plt.show()


# endregion


# region NewYork Airbnb

df = pd.read_csv(filepath_or_buffer='data/NewYork_City_ Airbnb.csv')
df.drop(['id', 'latitude', 'longitude', 'calculated_host_listings_count'], axis=1, inplace=True)


print(df.isnull().sum())  
df['last_review'].fillna('No Review', inplace=True)
df['reviews_per_month'].fillna(df['reviews_per_month'].mean(), inplace=True)

print(df.head().to_string())


# Kolon isimlerini düzenleme (room_type -> Room Type)
df.columns = df.columns.str.title()
print(df.columns)


# Her mahalle ("neighbourhood") için ortalama fiyatı hesaplama 
print(df.groupby('Neighbourhood')['Price'].mean())


# price sütunundaki değerleri z-score standardizasyonu ile dönüştürme
# scaler = StandardScaler()
# df['Price_Zscore'] = scaler.fit_transform(df[['Price']])

# print(df[['Price', 'Price_Zscore']].head())


# Her mahalledeki ilan sayısını gösteren bir bar grafiği oluşturma
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='Neighbourhood', order=df['Neighbourhood'].value_counts().index, palette='viridis')
plt.xticks(rotation=90)
plt.title('Number of Listings in Each Neighbourhood', fontsize=16)
plt.xlabel('Neighbourhood', fontsize=14)
plt.ylabel('Number of Listings', fontsize=14)
plt.tight_layout()
plt.show()

# endregion 

