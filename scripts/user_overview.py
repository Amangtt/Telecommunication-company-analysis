import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import os
import psycopg2
df= pd.read_csv('C:/Users/hello/Desktop/Data/Copy of Week2_challenge_data_source(CSV).csv')
load_dotenv()
DB_HOST=os.getenv('DB_HOST')
DB_PORT=os.getenv('DB_PORT')
DB_NAME=os.getenv('DB_NAME')
DB_USER=os.getenv('DB_USER')
DB_PASSWORD=os.getenv('DB_PASSWORD')
class overview:
    def load_data(query):
        """con=psycopg2.connect(
            host=DB_HOST,
            port= DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        df=pd.read_sql_query(query,con)
        con.close()"""
        df=pd.read_csv(query)
        return df
    
    
        # Dataset preprocessing
    def handle_dataset(df):
    
        df['Total_DLUL_netflix']= df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
        df['Total_DLUL_Youtube']= df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
        df['Total_DLUL_Gaming']= df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
        df['Total_DLUL_social']= df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
        df['Total_DLUL_email']= df['Email DL (Bytes)'] + df['Email UL (Bytes)']
        df['Total_DLUL_Other']= df['Other DL (Bytes)'] + df['Other UL (Bytes)']
        df['Total_DLUL_google']= df['Google DL (Bytes)'] + df['Google UL (Bytes)']

        df_edited=df.drop(['DL TP < 50 Kbps (%)','50 Kbps < DL TP < 250 Kbps (%)',
        '250 Kbps < DL TP < 1 Mbps (%)',
        'DL TP > 1 Mbps (%)',
        'UL TP < 10 Kbps (%)',
        '10 Kbps < UL TP < 50 Kbps (%)',
        '50 Kbps < UL TP < 300 Kbps (%)',
        'UL TP > 300 Kbps (%)',
        'Nb of sec with 125000B < Vol DL',
        'Nb of sec with 1250B < Vol UL < 6250B',
        'Nb of sec with 31250B < Vol DL < 125000B',
        'Nb of sec with 37500B < Vol UL',
        'Nb of sec with 6250B < Vol DL < 31250B',
        'Nb of sec with 6250B < Vol UL < 37500B',
        'Nb of sec with Vol DL < 6250B',
        'Nb of sec with Vol UL < 1250B',
        'Netflix DL (Bytes)',
        'Netflix UL (Bytes)',
        'Youtube DL (Bytes)',
        'Youtube UL (Bytes)',
        'Gaming DL (Bytes)',
        'Gaming UL (Bytes)',
        'Social Media DL (Bytes)',
        'Social Media UL (Bytes)',
        'Other DL (Bytes)',
        'Other UL (Bytes)',
        'Google DL (Bytes)',
        'Google UL (Bytes)',
        'Email UL (Bytes)',
        'Email DL (Bytes)'], axis=1)

        
        columns_to_fill = [ 'IMEI',  'Last Location Name','IMSI']
        df_edited[columns_to_fill] = df[columns_to_fill].fillna(0)
        #For string
        str_fill=['Handset Manufacturer', 'Handset Type']
        df_edited[str_fill]= df_edited[str_fill].fillna('Not Given')
        
        #For number to fill with mean
        num_fill=['Avg RTT DL (ms)', 'Avg RTT UL (ms)','TCP DL Retrans. Vol (Bytes)','TCP UL Retrans. Vol (Bytes)','HTTP DL (Bytes)','HTTP UL (Bytes)',]
        for col in num_fill:
            df_edited[col] = df_edited[col].fillna(df_edited[col].mean())
        
        df_new = df_edited.dropna(subset=['MSISDN/Number','Bearer Id']).reset_index()
        null_count=df_edited.isnull().sum()
        
        return df_new
    
    # This function tells us the top 10 handset types and top 3 handset manufacturers
    def handset(df):
        handset= df['Handset Type'].value_counts()
        top10=handset.head(10)
        
        handset= df['Handset Manufacturer'].value_counts()
        top3=handset.head(3)
        return top10,top3
    # This function aggregates the users information
    def aggregation_per_user(df):
        
        aggregated_data = df.groupby('Bearer Id').agg(
        number_of_sessions=('Bearer Id', 'nunique'),          
        total_session_duration=('Dur. (ms)', 'sum'),     
        total_download_data=('Total DL (Bytes)', 'sum'),          
        total_upload_data=('Total UL (Bytes)', 'sum'),               
        total_data_volume_netflix=('Total_DLUL_netflix', 'sum'),
        total_data_volume_youtube=('Total_DLUL_Youtube', 'sum'),
        total_data_volume_Gaming=('Total_DLUL_Gaming', 'sum'),
        total_data_volume_social=('Total_DLUL_social', 'sum'),
        total_data_volume_google=('Total_DLUL_google', 'sum'),     
        total_data_volume_email=('Total_DLUL_email', 'sum'),
        total_data_volume_others=('Total_DLUL_Other', 'sum'),                         
        ).reset_index()
        return aggregated_data
    
        
    # This function segements the users into 10 classes
    def user_segementation(df_edited):
        df_edited['Total Data']= df_edited['Total UL (Bytes)'] + df_edited['Total DL (Bytes)']

        df_edited['Decile'] = pd.qcut(df_edited['Dur. (ms)'], 10, labels=False,  duplicates='drop') + 1
        decile_totals  = df_edited.groupby('Decile').agg({
            'Dur. (ms)': 'sum',
            'Total Data': 'sum'
            }).reset_index()
        decile_totals=decile_totals.sort_values(by= 'Dur. (ms)')
        return decile_totals
    


    # This function computes Non-Graphical Univariate Analysis
    def compute_dispersion(df):
        # Drop excluded columns
        exclude_columns = ['Bearer Id', 'Start ms','Start','End','Last Location Name','Handset Manufacturer','Handset Type', 'End ms', 'Dur. (ms)', 'IMSI', 'MSISDN/Number', 'IMEI']
        df_filtered = df.drop(columns=exclude_columns)
        
        dispersion_summary = pd.DataFrame()
        
        # Compute statistics for each numeric column
        for column in df_filtered.select_dtypes(include='number').columns:
            mean = df_filtered[column].mean()
            median = df_filtered[column].median()
            variance = df_filtered[column].var()
            std_dev = df_filtered[column].std()
            range_val = df_filtered[column].max() - df_filtered[column].min()
            iqr = df_filtered[column].quantile(0.75) - df_filtered[column].quantile(0.25)

            dispersion_summary[column] = [mean, median, variance, std_dev, range_val, iqr]

        dispersion_summary.index = ['Mean', 'Median', 'Variance', 'Standard Deviation', 'Range', 'Interquartile Range']
        return dispersion_summary

    

   #Univariate analysis
    def uni(df):
        variances = {
        'Social': df['Total_DLUL_social'].var(),
        'Netflix': df['Total_DLUL_netflix'].var(),
        'YouTube': df['Total_DLUL_Youtube'].var(),
        'Gaming': df['Total_DLUL_Gaming'].var(),
        'Others': df['Total_DLUL_Other'].var(),
        'Email': df['Total_DLUL_email'].var(),
        'Google': df['Total_DLUL_google'].var()
        }
        col=['Total_DLUL_netflix','Total_DLUL_Youtube','Total_DLUL_Gaming','Total_DLUL_social','Total_DLUL_email','Total_DLUL_Other','Total_DLUL_google']
        
        for app, color in zip(col, ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']):
            plt.hist(df[app], bins=20, color=color, alpha=0.5, edgecolor='black')

        # Add labels and title
        plt.xlabel('Total variance')
        plt.ylabel('Frequency')
        plt.title('Univariate analysis')

        # Annotate the variances with vertical lines
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
        for (app, variance), color in zip(variances.items(), colors):
            plt.axvline(x=variance, color=color, linestyle='dashed', linewidth=1, label=f'{app} Variance: {variance:.2e}')
        plt.legend()
        plt.show()
    
        
    #Bivariate analysis
    def bi(df):
    
        df['Total_DL']= df['Netflix DL (Bytes)'] + df['Gaming DL (Bytes)'] + df['Social Media DL (Bytes)']+df['Email DL (Bytes)']+df['Other DL (Bytes)'] + df['Google DL (Bytes)']+ df['Youtube DL (Bytes)']
        df['Total_UL']= df['Netflix UL (Bytes)'] + df['Gaming UL (Bytes)'] + df['Social Media UL (Bytes)']+df['Email UL (Bytes)']+df['Other UL (Bytes)'] + df['Google UL (Bytes)']+ df['Youtube UL (Bytes)']
        df['Total_DLUL']= df['Total_UL'] + df['Total_DL']
        
        
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        scat=['Total_DLUL_netflix','Total_DLUL_Youtube','Total_DLUL_Gaming','Total_DLUL_social','Total_DLUL_email','Total_DLUL_Other','Total_DLUL_google']
        # Create scatter plots
        for ax, scat in zip(axes, scat): 
            sns.scatterplot(x=df[scat], y=df['Total_DLUL'], ax=ax)
            
            ax.set_xlabel(scat)
            ax.set_ylabel('Total DL + UL (Bytes)')

        plt.tight_layout()
        plt.show()


    #Correlation analysis
    def correlation(df_edited):

        df=df_edited[['Total_DLUL_netflix','Total_DLUL_Youtube','Total_DLUL_Gaming','Total_DLUL_social','Total_DLUL_email','Total_DLUL_Other','Total_DLUL_google']]
        
        correlation_matrix = df.corr()
       
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.show()
    

    #principal component analysis
    def PCAA(df):
        exclude_columns = ['Bearer Id','Start ms','End ms', 'Start','End','Last Location Name','Handset Manufacturer','Handset Type', 'Dur. (ms)', 'IMSI', 'MSISDN/Number', 'IMEI']
        df_filtered = df.drop(columns=exclude_columns)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered)
        pca = PCA()
        pca.fit(scaled_data)

        # Explained variance
        explained_variance = pca.explained_variance_ratio_

        # Plot the explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.title('Variance by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel(' Variance Ratio')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.grid()
        plt.show()
        
    

