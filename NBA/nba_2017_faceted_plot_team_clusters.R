library('readr')
library('ggplot2')

team_cluster <- read_csv('C://Users/Julie/nba_2017_att_val_elo_win_housing_cluster.csv', 
                         col_types = cols(X1 = col_skip()))

# Name clusters
team_cluster$cluster_name[team_cluster$cluster == 0] <- "High Valuation / Low Performance"
team_cluster$cluster_name[team_cluster$cluster == 1] <- "Medium Valuation / High Performance"
team_cluster$cluster_name[team_cluster$cluster == 2] <- "Low Valuation / Low Performance"

# Create a faceted plot
p <- ggplot(data = team_cluster) +
    geom_point(mapping = aes(x = ELO,
                             y = VALUE_MILLIONS,
                             color = factor(WINNING_SEASON, 
                                            labels = c("LOSING", "WINNING")),
                             size = MEDIAN_HOME_PRICE_COUNTY_MILLIONS,
                             shape = CONF)) +
    facet_wrap(~ cluster_name) +
    ggtitle("NBA Teams 2016-2017 Faceted Plot of Valuation of Team and Performance (ELO)") +
    ylab("Value NBA Team in Millions") +
    xlab("Relative Team Performance (ELO)") +
    geom_text(aes(x = ELO, 
                  y = VALUE_MILLIONS,
                  label = ifelse(VALUE_MILLIONS > 1200,
                                 as.character(TEAM), '')),
              hjust = 0.35,
              vjust = 1)

# Change legends
p + guides(color = guide_legend(title = "Winning Season")) +
    guides(size = guide_legend(title = "Median Home Price County in Millions")) +
    guides(shape = guide_legend(title = "NBA Conference"))
