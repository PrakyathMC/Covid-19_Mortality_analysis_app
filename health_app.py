import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import missingno as msno
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv('final_df.csv')

st.title("Disease Mortality Rate App")
st.sidebar.info("Welcome to my Mortality Analysis App")
st.sidebar.info("Created by [Prakyath Mannungal Chandran]")

def main():
    option = st.selectbox('Select an option', ["Statistical Analysis","Visual Representation","Prediction"])
    if option == 'Statistical Analysis':
        statistical_analysis()
    elif option == 'Visual Representation':
        visual_representation()
    else:
        prediction()


def statistical_analysis():
    st.write("This section provides insights to the data")
    st.dataframe(df.head(5))
    st.dataframe(df.tail(5))
    st.write("DataFrame shape is", {df.shape})

    st.write("Missing values summary:")
    if st.dataframe(df.isnull().sum() == 0):
        st.write(f"The number of missing values are zero")
    else:
        st.write(f"The dataset includes missing values")

    st.markdown("<span style='color:green'>Note: Certain coins will have lesser rows compared to Bitcoin,"
                " Since many coins were recently introduced in the market</span>", unsafe_allow_html=True)

    st.write("Columns in the dataset are:")
    st.dataframe(df.columns)
    st.write(f"Summary Statistics of the dataset are:")
    st.dataframe(df.describe())
    st.write("Outlier Detection")
    st.markdown("<span style='color:Orange'>Note: Lower and Upper bounds identified, </span>", unsafe_allow_html=True)

    def outliers():
        numerical_columns = [
            'Total Deaths',
            'COVID-19 Deaths',
            'Influenza Deaths',
            'Pneumonia Deaths',
            'Pneumonia and COVID-19 Deaths',
            'Pneumonia, Influenza, or COVID-19 Deaths'
        ]

        for column in numerical_columns:
            IQR = df[column].quantile(0.75) - df[column].quantile(0.25)

            lower_bridge = df[column].quantile(0.25) - (IQR * 3)
            upper_bridge = df[column].quantile(0.75) + (IQR * 3)

            st.markdown(f"<span style='color:green'>{column}</span>", unsafe_allow_html=True)

            st.write(f"Lower boundary: {lower_bridge}")
            st.write(f"Upper boundary: {upper_bridge}\n\n")

    # Call the outliers function after defining it
    outliers()

def visual_representation():
    option = st.radio('Choose an option to see the visualization:', [
        'Percentage of zeros',
        'Missing Values in data',
        'Data-types',
        'Outlier Detection',
        'Months with most mortality',
        'Correlation plots',
        'Hierarchical Clustering of Deaths with States',
        'Number of Deaths per state',
        'Top-10 cities with most deaths',
        'Growth/Trend of different deaths with time',
        'Deaths compared to Place of Death',
        'Time Series with highest and lowest death rate'
    ])

    def zeros():
        # Calculate the percentage of zero values in each column
        percentage_zeros = (df == 0).sum() / len(df) * 100
        percentage_non_zeros = 100 - percentage_zeros

        # Create subplots with pie charts
        num_columns = len(df.columns)
        num_rows = (num_columns + 1) // 2

        fig, axs = plt.subplots(num_rows, 2, figsize=(13, num_rows * 2))

        # Custom colors for the column names and pie chart labels
        column_name_color = 'blue'
        label_colors = ['red', 'green']

        for index, (column, ax) in enumerate(zip(df.columns, axs.flatten())):
            ax.pie([percentage_zeros[column], percentage_non_zeros[column]], labels=['Zero Values', 'Non-Zero Values'],
                   autopct='%1.1f%%', startangle=90, colors=label_colors)
            ax.set_title(column, color=column_name_color)

        # Adjust layout and remove unused subplots (if any)
        fig.tight_layout()
        for unused_ax in axs.flatten()[index + 1:]:
            unused_ax.axis('off')

        plt.suptitle('Percentage of Zero Values in Each Column of the Sampled DataFrame', y=1.02, fontsize=16)
        st.pyplot(fig)


    def missing_values():
        matplotlib.rcParams['figure.figsize'] = (14, 6)
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        st.pyplot(fig)

    def data_types():

        fig, ax = plt.subplots(figsize=(8,6))
        dtype_count = df.dtypes.value_counts()
        # bar plot
        dtype_count.plot(kind='bar', ax=ax)
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Number of columns')
        ax.set_title('Number of Columns by Data Types')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Pass the Matplotlib figure to Streamlit
        st.pyplot(fig)

    def outliers():
        # box plot
        fig1, axes = plt.subplots(3,3,sharex=True)
        df.plot(kind='box', subplots=True, layout=(3, 3), sharex=True)
        st.pyplot(fig1)

        #density plots
        numeric_columns = [
            'Year',
            'Month',
            'COVID-19 Deaths',
            'Total Deaths',
            'Pneumonia Deaths',
            'Pneumonia and COVID-19 Deaths',
            'Influenza Deaths',
            'Pneumonia, Influenza, or COVID-19 Deaths'
        ]
        fig2, axes = plt.subplots(nrows=1, ncols=len(numeric_columns), figsize=(15, 5))

        for i, column in enumerate(numeric_columns):
            sns.kdeplot(data=df[column], ax=axes[i], fill=True)
            # axes.set_title(f'Density Plot of {column}')

        plt.tight_layout()
        st.pyplot(fig2)

        #histogram plots
        fig3 = make_subplots(rows=1, cols=len(numeric_columns), subplot_titles=numeric_columns,
                            column_widths=[0.3] * len(numeric_columns))

        for i, column in enumerate(numeric_columns):
            fig3.add_trace(go.Histogram(x=df[column], nbinsx=30), row=1, col=i + 1)

        fig3.update_layout(
            title='Histograms of Numeric Columns',
            width=1200,
            height=500,
            margin=dict(l=10, r=10, t=175, b=10),
            showlegend=False
        )

        for annotation in fig3['layout']['annotations']:
            annotation['textangle'] = 45
        st.plotly_chart(fig3)

    def months_mortality():
        fig, ax = plt.subplots()
        fig = sns.jointplot(x='Total Deaths', y='Month', data=df, kind='scatter')
        st.pyplot(fig)

    def corrlation_plots():
        fig1, ax = plt.subplots()
        sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
        st.pyplot(fig1)

        fig2, ax = plt.subplots()
        sns.heatmap(df.corr(), cmap='magma', linecolor='white', linewidths=1, annot=True)
        st.pyplot(fig2)


    def clustering():
        fig, ax = plt.subplots(figsize=(12, 8))
        statewise_sum = df.pivot_table(
        values=['COVID-19 Deaths', 'Pneumonia Deaths', 'Pneumonia and COVID-19 Deaths', 'Influenza Deaths',
                    'Pneumonia, Influenza, or COVID-19 Deaths', 'Total Deaths'], index='State', aggfunc=np.sum)

        scaler = MinMaxScaler()
        statewise_normalised = pd.DataFrame(scaler.fit_transform(statewise_sum), index=statewise_sum.index,
                                            columns=statewise_sum.columns)

        # Perform hierarchical clustering
        linkage_matrix = linkage(statewise_normalised, method='ward')

        # Create the dendrogram
        dendrogram(linkage_matrix, labels=statewise_normalised.index, leaf_rotation=90)
        ax.set_xlabel('State')
        ax.set_ylabel('Distance (Ward)')
        ax.set_title('Hierarchical Clustering Dendrogram for States')
        st.pyplot(fig)

    def deaths_state():
        statewise = pd.pivot_table(df, values=['COVID-19 Deaths', 'Pneumonia Deaths', 'Pneumonia and COVID-19 Deaths',
                                               'Influenza Deaths', 'Pneumonia, Influenza, or COVID-19 Deaths',
                                               'Total Deaths'], index='State', aggfunc=np.sum)
        statewise = statewise.sort_values(by=['Total Deaths'], ascending=False)
        st.write(statewise.style.background_gradient(cmap='cubehelix'))

    def top_ten():
        fig,ax = plt.subplots(figsize=(16, 9))
        top_10_cities = df.groupby(by='State').max()[
            ['COVID-19 Deaths', 'Pneumonia Deaths', 'Pneumonia and COVID-19 Deaths', 'Influenza Deaths',
             'Pneumonia, Influenza, or COVID-19 Deaths', 'Total Deaths', 'Start Date']].sort_values(by=['Total Deaths'],
                                                                                                    ascending=False).reset_index()
        plt.title("Top 10 states with most mortality", size=25)
        ax = sns.barplot(data=top_10_cities.iloc[:10], y="Total Deaths", x="State", linewidth=2, edgecolor='black')
        plt.xlabel('States')
        plt.ylabel('Total Mortality')
        st.pyplot(fig)

    def trend():
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['End Date'] = pd.to_datetime(df['End Date'])

        grouped_start_data = df.groupby('Start Date').agg({
            'COVID-19 Deaths': 'sum',
            'Pneumonia Deaths': 'sum',
            'Influenza Deaths': 'sum'
        })

        grouped_end_data = df.groupby('End Date').agg({
            'COVID-19 Deaths': 'sum',
            'Pneumonia Deaths': 'sum',
            'Influenza Deaths': 'sum'
        })

        cumulative_df_start = grouped_start_data.cumsum()
        cumulative_df_end = grouped_end_data.cumsum()

        cumulative_df_start.reset_index(inplace=True)
        cumulative_df_end.reset_index(inplace=True)

        melted_df_start = cumulative_df_start.melt(id_vars='Start Date', var_name='Death Type',
                                                   value_name='Cumulative Deaths')
        melted_df_end = cumulative_df_end.melt(id_vars='End Date', var_name='Death Type',
                                               value_name='Cumulative Deaths')

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Start Date', 'End Date'))

        for death_type in melted_df_start['Death Type'].unique():
            fig.add_trace(
                go.Scatter(x=melted_df_start.loc[melted_df_start['Death Type'] == death_type, 'Start Date'],
                           y=melted_df_start.loc[melted_df_start['Death Type'] == death_type, 'Cumulative Deaths'],
                           name=death_type + ' (Start Date)'),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=melted_df_end.loc[melted_df_end['Death Type'] == death_type, 'End Date'],
                           y=melted_df_end.loc[melted_df_end['Death Type'] == death_type, 'Cumulative Deaths'],
                           name=death_type + ' (End Date)'),
                row=1, col=2
            )

        # Update the layout
        fig.update_layout(title='Growth Trend of Different Types of Deaths Based on Start Date and End Date',
                          showlegend=True)

        # Show the plot in the Streamlit app
        st.plotly_chart(fig)

    def place_death(df):
        fig, ax = plt.subplots()
        df = df[df['Place of Death'] != 'Total - All Places of Death']

        # Group by Place of Death and calculate the sum of COVID-19 deaths
        place_of_death_df = df.groupby('Place of Death').agg({'COVID-19 Deaths': 'sum'}).reset_index()

        # Create a pie chart
        fig = px.pie(place_of_death_df, values='COVID-19 Deaths', names='Place of Death',
                     title='COVID-19 Deaths by Place of Death')
        st.plotly_chart(fig)

    def time_series():
        state1 = 'Texas'
        state2 = 'District of Columbia'

        filtered_df = df[(df['State'] == state1) | (df['State'] == state2)]
        filtered_df['Start Date'] = pd.to_datetime(filtered_df['Start Date'])
        grouped_df = filtered_df.groupby(['Start Date', 'State']).agg({'COVID-19 Deaths': 'sum'}).reset_index()

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#eafff5')
        for state in [state1, state2]:
            state_data = grouped_df[grouped_df['State'] == state]
            ax.plot(state_data['Start Date'], state_data['COVID-19 Deaths'], label=state)

        ax.set_title(f'COVID-19 Deaths for {state1} and {state2}')
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('COVID-19 Deaths', fontsize=14)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        ax.legend()
        st.pyplot(fig)

    if option == 'Percentage of zeros':
        zeros()
    elif option == 'Missing Values in data':
        missing_values()
    elif option == 'Data-types':
        data_types()
    elif option == 'Outlier Detection':
        outliers()
    elif option == 'Months with most mortality':
        months_mortality()
    elif option == 'Correlation plots':
        corrlation_plots()
    elif option == 'Hierarchical Clustering of Deaths with States':
        clustering()
    elif option == 'Number of Deaths per state':
        deaths_state()
    elif option == 'Top-10 cities with most deaths':
        top_ten()
    elif option == 'Growth/Trend of different deaths with time':
        trend()
    elif option == 'Deaths compared to Place of Death':
        place_death(df)
    elif option == 'Time Series with highest and lowest death rate':
        time_series()


def prediction():
    fig, ax = plt.subplots(figsize=(15, 8))
    X = df[['Year', 'Month', 'Total Deaths', 'Pneumonia Deaths', 'Influenza Deaths']]
    y = df['COVID-19 Deaths']

    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
    st.pyplot(fig)

    mse_val = mse(y_test,y_pred)
    st.write(f"Mean Squared Error: {mse_val}")
    st.write(f"R-Squared {r2_score(y_test,y_pred)}")


if __name__ == '__main__':
    main()