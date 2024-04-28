import streamlit as st
import pandas as pd
import numpy as np

#######################################################
################## Help Functions #####################
#######################################################
@st.cache_data
def get_sales_data() -> pd.DataFrame:
    try:
        df = pd.read_csv('clean_50k_sales.csv')
    except FileNotFoundError:
        st.warning('Varning! Kunde inte hitta data att ladda in!!!', icon="⚠️")
        return pd.DataFrame()
    return df


#######################################################
################## Different pages ####################
#######################################################
def intro():
    import streamlit as st

    st.write("# Välkommen till Dashboard Party! 🎉")

    image_path = "./DashboardParty.webp"
    st.image(image_path,caption='Dashboard Party')
    st.markdown(
        """
        ## Vad är Dashboard Party?
        Dashboard Party är en app där du kan utforska allt roligt som kan göras med dashboards. Från interaktiva dataanalysverktyg till visuella representationer av intressant statistik.

        ### Funktioner
        - **Utforska Dashboards**: Dyk ner i olika dashboards och se vad du kan upptäcka.
        - **Skapa Egna**: Använd våra verktyg för att skapa dina egna dashboards.
        - **Dela Med Vänner**: Dela dina skapelser med andra och få feedback.

        ### Kom Igång
        För att komma igång, välj en kategori från sidomenyn eller skapa en ny dashboard direkt!

        ---

        Vi hoppas att du kommer ha lika roligt som vi hade när vi byggde denna app! Om du har några frågor eller feedback, tveka inte att kontakta oss.

        ---
    """
    )
    name = st.text_input("Skriv in ditt namn :raised_hand_with_fingers_splayed:", placeholder='Namn')
    if name:
        st.write(f"### Välkommen {name} till Dashboard Party! 🎉📈📊")

#######################################################
def hist_plot():
    import plotly.express as px
    import plotly.figure_factory as ff
    sales_data = get_sales_data()

    st.markdown(
        """ # Antal ordra per region
        """)
    fig = px.histogram(sales_data, x='Region')
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

    
    fig = px.histogram(sales_data, x='Units Sold', nbins=50, title='Histogram över Enheter Sålda',
                   labels={"Units Sold": "Enheter Sålda"})
    st.plotly_chart(fig)

    st.subheader('Fördelning av vinst')
    fig = ff.create_distplot([sales_data['Total Profit'].values], ['Total Profit'], bin_size=100000,
                        show_hist=True, show_rug=False)
    st.plotly_chart(fig)


    # selected_item_type = st.selectbox('Välj produkttyp:', sales_data['Item Type'].unique())
    # filtered_data = sales_data[sales_data['Item Type'] == selected_item_type]
    # fig = px.histogram(filtered_data, x='Unit Price', nbins=30, title=f'Histogram för Enhetspris - {selected_item_type}',
    #                 labels={"Unit Price": "Enhetspris"})
    # st.plotly_chart(fig)

#######################################################
def pandas_df():
    import streamlit as st
    df = get_sales_data()

    st.header('Visar pandas dataframe :sunglasses:', divider='rainbow')
    st.write(df)
    st.write(':green[Vi kan enkelt ladda in en pandas dataframe och visa den! Sortering på kolumner fungera från start! Testa!!]')

#######################################################
def scatter():
    import streamlit as st
    import plotly.express as px
    sales_data = get_sales_data()

    st.header('_Scatter_', divider='gray')

    sub_sales_data = sales_data.loc[(sales_data['Item Type'] == 'Beverages')]
    sub_sales_data = sub_sales_data[['Units Sold','Order ID','Order Priority']]

    st.scatter_chart(
        sub_sales_data.loc[:1000, :],
        x='Order ID',
        y='Units Sold',
        color='Order Priority')

    fig = px.scatter(
        sales_data.loc[:50, :],
        x="Order ID",
        y="Units Sold",
        size="Total Profit",
        color="Region",
        hover_name="Country",
        # log_x=True,
        size_max=60,
    )
    st.subheader('Plotly', divider='violet')
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])

    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)

    
    fig = px.scatter(sales_data, x='Units Sold', y='Total Profit', color='Country', title='Total Vinst mot Enheter Sålda per Land',
                    labels={"Units Sold": "Enheter Sålda", "Total Profit": "Total Vinst"},
                    hover_data=['Country'])

    st.plotly_chart(fig)

 
    fig = px.scatter(sales_data, x='Unit Cost', y='Unit Price', color='Item Type', title='Enhetspris mot Enhetskostnad per Produkttyp',
                    labels={"Unit Cost": "Enhetskostnad", "Unit Price": "Enhetspris"},
                    hover_data=['Item Type'])

    st.plotly_chart(fig)

 
    selected_region = st.selectbox('Välj region:', sales_data['Region'].unique())
    filtered_data = sales_data[sales_data['Region'] == selected_region]

    fig = px.scatter(filtered_data, x='Total Cost', y='Total Profit', color='Item Type', title=f'Scatter Plot för {selected_region}: Total Kostnad mot Total Vinst',
                    labels={"Total Cost": "Total Kostnad", "Total Profit": "Total Vinst"},
                    hover_data=['Item Type', 'Country'])

    st.plotly_chart(fig)

#######################################################
def sales():
    import streamlit as st
    import datetime
    sales_data = get_sales_data()
    min_date = datetime.datetime(2010,1,31)
    max_date = datetime.datetime(2020,9,30)

    col1,col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            'Startdatum',
            value=min_date,
            min_value=min_date,
            max_value=max_date
            )
    with col2:
        end_date = st.date_input(
            'Slutdatum',
            value=max_date,
            min_value=min_date,
            max_value=max_date
            )
    
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
    end_date = datetime.datetime(end_date.year, end_date.month,end_date.day)

    sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])
    sales_data = sales_data.loc[(sales_data['Order Date'] > start_date)&(sales_data['Order Date'] < end_date)]

    date_sales = sales_data.set_index('Order Date')['Total Revenue'].resample('ME').sum()

    st.line_chart(date_sales)

    if st.checkbox('Visa dataframe'):
        date_sales

#######################################################
def filter_():
    import streamlit as st 
    sales_data = get_sales_data()
    region = st.selectbox('Välj region:', sales_data['Region'].unique()) 

    if st.checkbox('Sortera på land'):
        country_in_region = sales_data.loc[sales_data['Region'] == region]

        country = st.selectbox('Välj land:', country_in_region['Country'].unique())

        filtered_data = sales_data[(sales_data['Region'] == region) & (sales_data['Country'] == country)]
    else:
        filtered_data = sales_data[sales_data['Region'] == region]

    st.write(filtered_data)

#######################################################
def test_map():
    import streamlit as st
    import numpy as np
    import pandas as pd

    NR_OF_DOTS = 200
    # 59.30971829478414, 18.02171685252132
    map_data = pd.DataFrame(
        np.random.randn(NR_OF_DOTS, 2) / [50, 50] + [59.31, 18.02],
        columns=['lat', 'lon'])
    
    # RGB
    c = [[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]]
    colors = [c[i] for i in np.random.randint(low=0,high=3,size=NR_OF_DOTS)]
    map_data['colors'] = colors
    
    st.header('_Olykor på stan_', divider='red')

    st.map(
        map_data,
        latitude='lat',
        longitude='lon', 
        size=10, 
        color='colors')

#######################################################
def heatmaps():
    import plotly.express as px
    import streamlit as st
    sales_data = get_sales_data()
    z = sales_data.select_dtypes(include=['float64','int64']).corr()
    fig = px.imshow(z, text_auto=True)

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)


#######################################################
def bar_chart():
    import streamlit as st
    import plotly.express as px

    sales_data = get_sales_data()

    st.header('Bar Charts Party', divider='rainbow')

    st.bar_chart(sales_data['Sales Channel'].value_counts())

    st.subheader('Ordrar per land!', divider='green')

    st.bar_chart(sales_data['Country'].value_counts())


    top_countries_profit = sales_data.groupby('Country')['Total Profit'].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_countries_profit, 
        x='Country', 
        y='Total Profit', 
        title='Total Vinst per Land', 
        color='Total Profit')
    
    st.plotly_chart(fig)

    units_sold_per_item = sales_data.groupby('Item Type')['Units Sold'].sum().reset_index()
    fig = px.bar(
        units_sold_per_item, 
        x='Item Type', 
        y='Units Sold', 
        title='Antal Sålda Enheter per Produkttyp', 
        color='Units Sold')
    
    st.plotly_chart(fig)

    average_price_per_item = sales_data.groupby('Item Type')['Unit Price'].mean().reset_index()
    fig = px.bar(
        average_price_per_item, 
        x='Item Type', 
        y='Unit Price', 
        title='Genomsnittligt Enhetspris per Produkttyp', 
        color='Unit Price')
    
    st.plotly_chart(fig)

#######################################################
def pie_chart():
    import streamlit as st
    import plotly.express as px
    sales_data = get_sales_data()
    

    product_performance = sales_data.groupby('Item Type')['Total Revenue'].sum().reset_index()

    fig = px.pie(
        product_performance,
        values='Total Revenue',
        names='Item Type',
        title='Total försäljning per produkt!')

    st.plotly_chart(fig)

    channel_revenue = sales_data.groupby('Sales Channel')['Total Revenue'].sum().reset_index()
    priority_revenue = sales_data.groupby('Order Priority')['Total Revenue'].sum().reset_index()


    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(channel_revenue, values='Total Revenue', names='Sales Channel', title='Försäljning per Kanal')
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig = px.pie(priority_revenue, values='Total Revenue', names='Order Priority', title='Försäljning per Orderprioritet')
        st.plotly_chart(fig,use_container_width=True)


    region_revenue = sales_data.groupby('Region')['Total Revenue'].sum().reset_index()
    fig = px.pie(region_revenue, values='Total Revenue', names='Region', title='Försäljning per Region')
    st.plotly_chart(fig,use_container_width=True)

#######################################################
##################### Page setup ######################
#######################################################

page_names_to_funcs = {
    "Start": intro,
    "Histogram":hist_plot,
    "Pandas":pandas_df,
    "Scatter":scatter,
    "Sales":sales,
    "Filter":filter_,
    "Map":test_map,
    "HeatMap":heatmaps,
    "Bar Chart":bar_chart,
    "Pie Chart": pie_chart


}

demo_name = st.sidebar.selectbox("Välj en Dashboard", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
