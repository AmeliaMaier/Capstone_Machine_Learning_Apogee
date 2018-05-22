
def update_urls_table_from():
    return '''
            INSERT INTO urls (url_raw, site_description, html_raw, linked, root_url)
                VALUES (%(url_raw)s, %(site_description)s, %(html_raw)s, %(linked)s, %(root_url)s)
                ON CONFLICT (url_raw)
                DO UPDATE SET (site_description, html_raw, linked, root_url)
                    =(EXCLUDED.site_description, EXCLUDED.html_raw, EXCLUDED.linked, EXCLUDED.root_url);
            '''

def update_urls_table_tos(links):
    sql_statement = ' INSERT INTO urls (url_raw) VALUES '
    link_vars = []
    link_string_urls = ' '
    for ind in range(len(links)):
        if ind == 0:
            link_string_urls += f'( %({"link" + str(ind)})s )'
            link_vars.append(f'{"link" + str(ind)}')
        else:
            link_string_urls += f', ( %({"link" + str(ind)})s )'
            link_vars.append(f'{"link" + str(ind)}')
    var_dict_addition = dict(zip(link_vars, links))
    sql_statement += link_string_urls
    sql_statement += ' ON CONFLICT (url_raw) DO NOTHING; '

    #add links to website_links
    sql_statement += ' INSERT INTO website_links (from_url_ID, to_url_ID) VALUES '
    link_string_website_links = ' '
    for ind in range(len(link_vars)):
        if ind == 0:
            link_string_website_links += '((SELECT url_ID FROM urls WHERE url_raw = %(url_raw)s), (SELECT url_ID FROM urls WHERE url_raw = '
            link_string_website_links += f'%({link_vars[ind]})s))'
        else:
            link_string_website_links += ', ((SELECT url_ID FROM urls WHERE url_raw = %(url_raw)s), (SELECT url_ID FROM urls WHERE url_raw = '
            link_string_website_links += f'%({link_vars[ind]})s))'
    sql_statement += link_string_website_links
    sql_statement += ' ON CONFLICT ON CONSTRAINT website_links_pkey DO NOTHING;'
    return sql_statement, var_dict_addition

def check_url_already_seen():
    return '''
            SELECT COUNT(*) FROM urls
                WHERE urls.url_raw = %(url_raw)s
                AND urls.linked = %(linked)s;
            '''

def bulk_check_url_already_seen():
    return '''
            SELECT url_raw FROM urls
                WHERE urls.linked = %(linked)s;
            '''

def get_url_layer():
    return '''
            SELECT urls.url_raw FROM urls
                LEFT JOIN website_links
                    ON urls.url_ID = website_links.from_url_ID
                WHERE NOT urls.linked
                AND website_links.from_url_ID is NULL;
            '''
