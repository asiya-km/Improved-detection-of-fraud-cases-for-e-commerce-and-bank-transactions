def get_geolocation(ip_address, ip_to_country):
    try:
        ip_address = int(ip_address)
        country = ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip_address) & 
                                (ip_to_country['upper_bound_ip_address'] >= ip_address)]['country'].iloc[0]
    except (ValueError, IndexError):
        country = 'Unknown'
    return country 