import web_link_mapping as wlm



if __name__='__main__':
    'Remember to create database first. Code in create_db_local_notes.sql'
    urls_starting_points = pickle.load(open("data/url_series.p","rb"))
    urls_starting_points = urls_starting_points - wlm.bulk_check_url_already_seen()
    urls_starting_points = list(urls_starting_points)
    wlm.initial_load_threaded(urls_starting_points)
    wlm.crawl_thread(depth=2)
