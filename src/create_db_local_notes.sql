$ psql

	CREATE DATABASE website_link_mapping;

$ psql website_link_mapping

	CREATE TABLE urls (
	url_ID SERIAL PRIMARY KEY,
	url_raw VARCHAR NOT NULL,
	root_url VARCHAR DEFAULT 'not_available',
	site_description VARCHAR DEFAULT 'not_available',
	date_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	);
	CREATE TABLE website_links (
	from_url_ID INTEGER NOT NULL REFERENCES urls(url_id),
	to_url_ID INTEGER NOT NULL REFERENCES urls(url_id),
	website_link_ID SERIAL PRIMARY KEY
	);
