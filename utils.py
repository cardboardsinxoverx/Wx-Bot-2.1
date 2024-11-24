import datetime as dt

def parse_date(date):
    return dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
