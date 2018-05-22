CREATE DATABASE website_link_mapping;

CREATE TABLE urls (
url_ID SERIAL PRIMARY KEY,
url_raw VARCHAR NOT NULL,
root_url VARCHAR DEFAULT 'not_available',
site_description VARCHAR DEFAULT 'not_available',
date_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
html_raw VARCHAR,
linked BOOLEAN DEFAULT False
);
CREATE TABLE website_links (
from_url_ID INTEGER NOT NULL REFERENCES urls(url_id),
to_url_ID INTEGER NOT NULL REFERENCES urls(url_id),
PRIMARY KEY (from_url_ID, to_url_ID)
);

ALTER TABLE urls ADD UNIQUE (url_raw);
