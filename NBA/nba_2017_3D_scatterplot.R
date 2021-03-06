library('scatterplot3d')
library('readr')

team_cluster <- read_csv('C://Users/Julie/nba_2017_att_val_elo_win_housing_cluster.csv', 
                         col_types = cols(X1 = col_skip()))

cluster_to_numeric <- function(column) {
    converted_column <- as.numeric(unlist(column))
    return(converted_column)
}

team_cluster$pcolor[team_cluster$cluster == 0] <- "red"
team_cluster$pcolor[team_cluster$cluster == 1] <- "blue"
team_cluster$pcolor[team_cluster$cluster == 2] <- "darkgreen"

s3d <- scatterplot3d(
    cluster_to_numeric(team_cluster["VALUE_MILLIONS"]),
    cluster_to_numeric(team_cluster["MEDIAN_HOME_PRICE_COUNTY_MILLIONS"]),
    cluster_to_numeric(team_cluster["ELO"]),
    color = team_cluster$pcolor,
    pch = 19,
    type = "h",
    lty.hplot = 2,
    main = "3D Scatterplot NBA Teams 2016-2017 : Value, Performance, Home Prices with Knn Clustering",
    zlab = "Team Performance (ELO)",
    xlab = "Value of Team in Millions",
    ylab = "Median Home Price County Millions"
)

s3d.coords <- s3d$xyz.convert(cluster_to_numeric(team_cluster["VALUE_MILLIONS"]),
                              cluster_to_numeric(team_cluster["MEDIAN_HOME_PRICE_COUNTY_MILLIONS"]),
                              cluster_to_numeric(team_cluster["ELO"]))

# Plot text
text(s3d.coords$x, s3d.coords$y, # x and y coordinates
     labels = team_cluster$TEAM, # text to plot
     pos = 4, cex = 0.6) # shrink text place to right of points 

# Add legend
legend("topleft", inset = 0.05, # loaction and inset
       bty = "n", cex = 0.5, # suppress legend box, shrink text 50%
       title = "Clusters",
       c("0", "1", "2"), 
       fill = c("red", "blue", "darkgreen"))
