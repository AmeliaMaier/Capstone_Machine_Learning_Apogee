
## Network Graphing of Links Between Webpages

#### Proprietary Information Concerns:

The analysis included in this project includes revenue and cost data. The examples given are examples and not the true values. They have been created to show the same relationships as the true data.

### Stack:

<img src='images/networkGraphingStack.png'>

### Objectives:

+ Analyze existing data to determine if the associative rule of graph theory holds true for this application.
+ Create a web crawler that will perform a breadth first search for links on sites users have been seen at and map possible browsing routes from those starting points.
+ Create lists of likely profitable and unprofitable urls Apogee hasn’t advertised on before, at least in training set.
+ Compare predictions to full data set to see if how accurate they are.

### EDA and Feature Engineering:

I feature engineered the direct and indirect profitability of each website users were seen at. Based on the high positive correlation between the direct and indirect measures, users who are seen at one profitable website are likely to go to other profitable websites.

| URL                   |  URL ID  |  Direct Profit  |  Indirect Profit  |  Direct Sales  |  Indirect Sales  | Times Seen  |
| --------------------- |:--------:| ---------------:| ----------------- |:--------------:| ----------------:| -----------:|
| www.mail.yahoo.com/   |  1952    |  1              |  24.6             |    255         |  7996            |  125635     |
| www.ebay.co.uk/       |  3747    |  .10            |  .40              |  9             |  111             |  6419       |
| www.zillow.com/       | 496173   | -0.00225        |  -0.0225          |  0             |  21              |  1730       |

<img src='images/url_profile_summary_corr_heatmap.png'>

### Webcraler Network Graphs:

The web-crawler was given 11,000 starting points and was stopped when the database contained about 4 million nodes with about
14 million edges. I broke the data into sub-graphs, finding paths that spread out from the starting points. Below is the graph starting at 1952. It links to 18 other starting points which were all either profitable or seen very few times and just below profitable.


<img src='images/starting_point_1952.png'>

Below is the graph starting at 3747. It is an example of a profitable url that the webcrawler didn’t search far enough to link to
another known url.

<img src='images/starting_point_3747.png'>

### Conclusion:

Based on the data analysis, users who are seen at a website that has historically been profitable for Apogee and their client show are likely to be seen at other sites that are also profitable. 

Conversely, users that are seen at websites that have historically been unprofitable for Apogee and their client are unlikely to go to websites that are profitable.

The webcrawler correctly identified websites linked to the starting website that have also been profitable, along with a list
of websites that haven’t been seen in the past and are potentially new locations to advertise on.

Due to the interconnectedness of root websites (ebay leads to more ebay sites), it would be useful to allow the webcrawler to run for a longer time, creating deeper graphs with more chances to step outside the root website it started in.

A list of sites to avoid and to focus on has been provided to Apogee and is being compared to the client’s full history.
