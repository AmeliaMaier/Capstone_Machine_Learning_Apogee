
## Network Graphing of Links Between Webpages

#### Proprietary Information Concerns:

The analysis included in this project includes revenue and cost data. The examples given are examples and not the true values. They have been created to show the same relationships as the true data.

### Stack:

<img src='images/networkGraphingStack.png'>

### Objectives:

+ Analyze existing data to determine if the associative rule of graph theory holds true for this application.

>>>Associative Rule: if person A knows person B and person C, person B and C are more likely to know each other than if they didn't know A. In this context, it means that if a user goes to multiple websites and we know that one of those websites is profitable, the other websites that person goes to are more likely to be profitable. Also, if a user goes to a website and that website has links, the person is more likely to be seen where the links lead than any randomly selected url. 

+ Create a web crawler that will perform a breadth first search for links on sites users have been seen at and map possible browsing routes from those starting points.
+ Create lists of likely profitable and unprofitable urls Apogee hasn’t advertised on before, at least in training set.
+ Compare predictions to full data set to see if how accurate they are.

### EDA and Feature Engineering:

The raw dataset I started with had interaction logs. This means each row in the dataset represented one add being shown to one user on one website. Each users could have multiple rows of data. Each website could be on multiple rows for one or more users. Cost and revenue are known per interaction, so I am able to determine the general profitability of each interaction, user, website..ect. For more indepth information about this, see the [EDA Project](https://github.com/AmeliaMaier/Capstone_Machine_Learning_Apogee/blob/refactor/edaREADME.md).

For this specific project, I feature engineered the direct and indirect profitability of each website users were seen at. I define direct profitability for a website to be the total revenue due to a conversion allocated to ad seen on that website minus the costs associated with that ad. An example of the aggregation is below:

| URL  |  Revenue  |  Cost  |  Conversion  |
| ---- | ---------:| ------ |:------:| 
| A    |  1        | .01    |  1     | 
| A    |  0        | .01    |  0     | 
| A    | .5        | .02    |  1     |  

| URL  |  Revenue  |  Cost  |  Direct Conversions  | Direct Profit  |  Times Seen | 
| ---- | ---------:| ------ |:------:|:------:|:------:| 
| A    |  1.5       | .03    |  2     | 1.47  |  3  |

I define indirect profitability for a website to be the total revenue for all users who have been seen at that site at least once minus the total cost for those same users minus the direct profit for the url in question. I remove the direct profits so that all that is remained is the profit from other urls seen by the same users that view the questioned url.

| User | URL | Revenue | Cost | Conversion |
| ---- | ---- | ------:| ------ |:------:| 
| 1   |  A    |  1     | .01   |  1       |
| 1   |  B    |  1.5     | .05   |  1       |
| 1   |  C    |  0     | .02   |  0       |
| 2   | A    |  0        | .01    |  0     | 
| 2   | A    | .5        | .02    |  1     |  
| 3   | B    | .5        | .02    |  1     |  

| User | Revenue | Cost | Conversion |
| ---- | ------:| ------ |:------:| 
| 1   |  2.5     | .08   |  2       |
| 2   | .5        | .03    |  1     |  

| URL  |  Direct Conversions  | Indirect Conversions | Direct Profit  | Indirect Profit |  Times Seen | 
| ---- |:------:|:------:|:------:|:------:|:------:| 
| A    |  2     |  1  | 1.47  |  1.42  |  3  |

Here is an example of the actual direct to indirecct proportions seen in the data:

| URL            |  URL ID  |  Direct Profit  |  Indirect Profit  |  Direct Conversion  |  Indirect Conversion  | Times Seen  |
| --------------------- |:--------:| ---------------:| ----------------- |:--------------:| ----------------:| -----------:|
| www.mail.yahoo.com/   |  1952    |  1              |  24.6             |    255         |  7996            |  125635     |
| www.ebay.co.uk/       |  3747    |  .10            |  .40              |  9             |  111             |  6419       |
| www.zillow.com/       | 496173   | -0.00225        |  -0.0225          |  0             |  21              |  1730       |

Based on the high positive correlation between the direct and indirect measures, users who are seen at one profitable website are likely to go to other profitable websites and users who go one unprofitable website are likely to go to other unprofitable websites.

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
