def cleaning(df):
    import pandas as pd
    df = df.drop(df.columns[[0,1,2,3,4,5,6,12,14,15,16,18,19,20,21,22,23,25]], axis=1) #drop unnecessary columns
    df = df.rename(columns={'Geplande.datum.tijd': 'Geplande_datum_tijd', 'Pay.pax' : 'Pay_pax' ,'Max....zitplaatsen': 'Max_zitplaatsen'}) 
    df = df.drop(['Geplande_datum_tijd', 'BookedPayPax', 'time_12', 'tz', 'Vluchtnummer', 'Vliegtuigregistratie'], axis = 1)
    df['Max_zitplaatsen'] = pd.to_numeric(df['Max_zitplaatsen'], errors='coerce')
    df = df[(df['Verkeerstype'] == "Departure") | (df['Verkeerstype'] == "Arrival")] #only keep Departures and Arrivals
    df = df[df['Pay_pax'].isna() == 0] #delete NaN rows
    df= df[df['Pay_pax'] != 0] #delete 0 rows
    df = df[(df['Maatschappij'] == "Departure") | (df['Verkeerstype'] == "Arrival")] #only keep Departures and Arrivals
    df = df[df['Maatschappij'].isna() == 0] #delete NaN rows
    df = df[df['Max_zitplaatsen'].isna() == 0]
    df = df.drop(['Verkeerstype'], axis = 1) #drop unnecessary columns
    df = df.copy()
    df['Maatschappij'] = df['Maatschappij'].str.replace(' ', '')
    df['Maatschappij'] = df['Maatschappij'].str.replace('.', '')
    df['Maatschappij'] = df['Maatschappij'].str.lower()
    df['Maatschappij'] = df['Maatschappij'].str.replace('klmcityho', 'klm')
    df['Maatschappij'] = df['Maatschappij'].str.replace('klmacco', 'klm')
    df['Maatschappij'] = df['Maatschappij'].str.replace('asl-a', 'asl-air')
    df['Maatschappij'] = df['Maatschappij'].str.replace('asl-airir', 'asl-air')
    df['Maatschappij'] = df['Maatschappij'].str.replace('bacityflyer', 'ba')
    df['Maatschappij'] = df['Maatschappij'].str.replace('britishairways', 'ba')
    df['Maatschappij'] = df['Maatschappij'].str.replace('britisha', 'ba')
    df['Maatschappij'] = df['Maatschappij'].str.replace('bai', 'ba')
    df['Maatschappij'] = df['Maatschappij'].str.replace('cityjetlt', 'cityjet')
    df['Maatschappij'] = df['Maatschappij'].str.replace('corendonairways', 'corendon')
    df['Maatschappij'] = df['Maatschappij'].str.replace('corendona', 'corendon')
    df['Maatschappij'] = df['Maatschappij'].str.replace('corendoneurope', 'corendon')
    df['Maatschappij'] = df['Maatschappij'].str.replace('excellentair', 'excellent')
    df['Maatschappij'] = df['Maatschappij'].str.replace('freebirdeurope', 'freebird')
    df['Maatschappij'] = df['Maatschappij'].str.replace('jetnetherlands', 'jetnether')
    df['Maatschappij'] = df['Maatschappij'].str.replace('koningklijk', 'klm')
    df['Maatschappij'] = df['Maatschappij'].str.replace('londonexecutiveaviation', 'londonexe')
    df['Maatschappij'] = df['Maatschappij'].str.replace('netjetseurope', 'netjetseu')
    df['Maatschappij'] = df['Maatschappij'].str.replace('netjetstr', 'netjetseu')
    df['Maatschappij'] = df['Maatschappij'].str.replace('northflying', 'northfly')
    df['Maatschappij'] = df['Maatschappij'].str.replace('pegasusai', 'pegasus')
    df['Maatschappij'] = df['Maatschappij'].str.replace('quickaircharter', 'quickair')
    df['Maatschappij'] = df['Maatschappij'].str.replace('shellaircraftltd', 'shellairc')
    df['Maatschappij'] = df['Maatschappij'].str.replace('vlmairlin', 'vlm')
    df[['hours', 'minutes']] = df['time_24'].str.split(':', 1, expand=True)
    df = df.drop('time_24', axis=1)
    counts = df['Maatschappij'].value_counts()
    df = df[~df['Maatschappij'].isin(counts[counts < 50].index)]
    y = df.Pay_pax
    X = df.drop('Pay_pax', axis=1)
    return X, y
