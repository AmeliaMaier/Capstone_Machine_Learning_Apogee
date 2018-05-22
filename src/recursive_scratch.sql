

CREATE TEMP TABLE limited_links AS
SELECT from_url_ID, to_url_ID	FROM (
	SELECT ROW_NUMBER() OVER(PARTITION BY from_url_ID) AS row, website_links.* FROM	website_links) AS temp_grouping
WHERE temp_grouping.row <= 2;

WITH RECURSIVE first_level_elements AS (
		(
		SELECT from_url_ID, to_url_ID, array[from_url_ID] AS link_path, 0 depth_limit FROM limited_links
			WHERE from_url_ID = %(starting_point)s
		)
		UNION
			SELECT nle.from_url_ID, nle.to_url_ID, (fle.link_path || nle.from_url_ID), fle.depth_limit+1 FROM first_level_elements as fle
				JOIN limited_links as nle
					ON fle.to_url_ID = nle.from_url_ID
			WHERE NOT (nle.from_url_ID = any(link_path))
				AND fle.depth_limit < 2
	)
	SELECT from_url_ID, to_url_ID, (link_path || to_url_ID)as link_path, depth_limit  from first_level_elements;




-- trying to limit how many loops are done (bredth)
	CREATE TEMP TABLE limited_links AS
	SELECT from_id, to_id	FROM (
		SELECT ROW_NUMBER() OVER(PARTITION BY from_id) AS row, links.* FROM	links) AS temp_grouping
	WHERE temp_grouping.row <= 2;

	WITH RECURSIVE first_level_elements AS (
		--non recursive term
		(
		SELECT from_id, to_id, array[from_id] AS link_path, 0 depth_limit FROM limited_links
			WHERE from_id = 1
		)
		UNION
		--recursive term
			SELECT nle.from_id, nle.to_id, (fle.link_path || nle.from_id), fle.depth_limit+1 FROM first_level_elements as fle
				JOIN limited_links as nle
					ON fle.to_id = nle.from_id
			WHERE NOT (nle.from_id = any(link_path))
				AND fle.depth_limit < 3
	)
	SELECT from_id, to_id, (link_path || to_id) as link_path, depth_limit from first_level_elements;









-- trying to limit how many loops are done (depth)
WITH RECURSIVE first_level_elements AS (
	--non recursive term
	(
	SELECT from_id, to_id, array[from_id] AS link_path, 0 depth_limit FROM links
		WHERE from_id = 1
	)
	UNION
	--recursive term
		SELECT nle.from_id, nle.to_id, (fle.link_path || nle.from_id), fle.depth_limit+1 FROM first_level_elements as fle
			JOIN links as nle
				ON fle.to_id = nle.from_id
		WHERE NOT (nle.from_id = any(link_path))
			AND fle.depth_limit < 3
		LIMIT 2
)
SELECT from_id, to_id, (link_path || to_id) as link_path, depth_limit from first_level_elements;







-- fixing problem with LOOPS

WITH RECURSIVE first_level_elements AS (
	--non recursive term
	(
	SELECT from_id, to_id, array[from_id] AS link_path FROM links
		WHERE from_id = 1
	)
	UNION
	--recursive term
		SELECT nle.from_id, nle.to_id, (fle.link_path || nle.from_id) FROM first_level_elements as fle
			JOIN links as nle
				ON fle.to_id = nle.from_id
		WHERE NOT (nle.from_id = any(link_path)
)
SELECT from_id, to_id, (link_path || to_id) as link_path from first_level_elements;





--adding path PROBLEM WITH LOOPS
WITH RECURSIVE first_level_elements AS (
	--non recursive term
	(
	SELECT from_id, to_id, array[from_id, to_id] AS link_path FROM links
		WHERE from_id = 10
	)
	UNION
	--recursive term
		SELECT nle.from_id, nle.to_id, (fle.link_path || nle.to_id) FROM first_level_elements as fle
			JOIN links as nle
				ON fle.to_id = nle.from_id
)
SELECT * from first_level_elements;








---first version

WITH RECURSIVE first_level_elements AS (
	--non recursive term
	(
	SELECT from_id, to_id FROM links
		WHERE from_id = 1
	)
	UNION
	--recursive term
		SELECT nle.from_id, nle.to_id FROM first_level_elements as fle
			JOIN links as nle
				ON fle.to_id = nle.from_id
)
SELECT * from first_level_elements;
