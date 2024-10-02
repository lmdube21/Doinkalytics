library(tidyverse)
library(lubridate)
library(catboost)
library(mlbench)

#pull in stadium lat long data
stad_lat_long<- read_csv("stadium_lat_long.csv") |>
  rename(home_team = Abbr) |>
  select(-Team)


#pull in Sam's elevation data
sam_data <- read_csv("SamData.csv") 

sam_data <- sam_data |>
  mutate(game_date = ymd(mdy(Date))) |>
  rename(home_team = Home.Abbr)
         
sam_data <- sam_data |> select(c(home_team, elevation, latitude, longitude)) |> filter(longitude < -60 & latitude >20) |> select(c(home_team, elevation)) |> distinct()



# Convert to y-m-d format


#load schedule, remove international games (because current weather API doesn't have data outside of US)
sked <- nflreadr::load_schedules(2021:2024)
international_stads = c("FRA00", "SAO00", "GER00", "LON01", "LON01", "LON00", "MEX00")
nonint_sked <- sked |>
  filter(!stadium_id %in% international_stads) |>
  mutate(
    datetime_et = ymd_hm(paste(gameday, gametime), tz = "America/New_York"),
    datetime_et =  ceiling_date(datetime_et, unit = "hour"),
    datetime_gmt = with_tz(datetime_et, tzone = "GMT"),
    datetime_gmt = format(datetime_gmt, "%Y-%m-%dT%H:%M:%S")) |>
    left_join(stad_lat_long, by = "home_team") |>
    left_join(sam_data, by = "home_team")  



#save data to get weather
write.csv(nonint_sked, "schedule_no_inter.csv")

#go to python file to pull weather data
#pull updated weather data back in 
weather_data <- read_csv('noninter_final.csv') |>
  select(c(game_id, temperature, windgusts, windspeed, roof, surface, precipitation, elevation, turf, indoor))

#pull play by play from this year
pbp_data <- nflfastR::load_pbp(2021:2024)


# Filter to get only field goal kicking plays
fg_plays <- pbp_data |>
  select(c(play_id, play_type, game_id, kicker_player_id, home_team, away_team, season_type, week, season, posteam, yardline_100, game_date, qtr, down, time, yrdln, ydstogo, field_goal_result, kick_distance, home_timeouts_remaining, away_timeouts_remaining, total_home_score, total_away_score, ep, epa, wp, wpa, kicker_player_name, kicker_player_id, home_coach, away_coach, stadium_id, game_stadium)) |>
  left_join(weather_data, by = "game_id")|>
  filter(play_type =="field_goal"& field_goal_result != "blocked" ) |>
  mutate(binary_result = ifelse(field_goal_result == "made", 1, 0))

# Group by season and calculate the mean of field_goal_result and the total count
season_stats <- fg_plays %>%
  group_by(season) %>%
  summarize(mean_field_goal = mean(binary_result, na.rm = TRUE),
            total_count = n())

# Define a scaling factor to scale the second y-axis
scaling_factor <- max(season_stats$mean_field_goal) / max(season_stats$total_count)

# Create the dual-axis bar chart
ggplot(season_stats, aes(x = factor(season))) +
  # First bar for mean field goal result
  geom_bar(aes(y = mean_field_goal), stat = "identity", fill = "steelblue") +
  # Second bar for total count, scaled by the scaling factor
  geom_bar(aes(y = total_count * scaling_factor), stat = "identity", fill = "orange", alpha = 0.5) +
  # Labels on the top of the mean field goal bars
  geom_text(aes(y = mean_field_goal, label = round(mean_field_goal, 2)), vjust = -0.5) +
  # Labels on the top of the total count bars (adjusted for scaling)
  geom_text(aes(y = total_count * scaling_factor, label = total_count), vjust = -0.5, color = "orange") +
  # Add the left y-axis for mean field goal result
  scale_y_continuous(name = "Mean Field Goal Result", sec.axis = sec_axis(~./scaling_factor, name = "Kick Attempts by Season")) +
  labs(title = "Mean Field Goal Result and Total Rows by Season",
       x = "Season") +
  theme_minimal()

modeling_data <- fg_plays |> 
  select(c(game_id, play_id, binary_result, elevation, temperature, precipitation, windgusts, windspeed, kick_distance, turf)) |>
  drop_na() |>
  distinct()

modeling_data$binary_result <- factor(modeling_data$binary_result)#,  levels = c(0, 1), labels = c("Miss", "Make") )



create_calibration_plot <- function(model, data, actual_col, bins = 20, confperc = 0.95) {
  data <- data |>
    mutate(pred = predict(model, data, type = "prob"),
           predicted_probs = pred$`1`,
           num_result = as.numeric({{actual_col}})-1)
           

  
  # Step 1: Get predicted probabilities
  data$bins <- cut(data$predicted_probs, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE)
  
  # Calculate observed proportion (mean of actuals) and predicted proportion per bin
  calibration_data <- data %>%
    group_by(bins) %>%
    summarise(
      mean_pred = mean(predicted_probs),
      mean_obs = mean(num_result),
      n = n()
    )
  
  # Calculate confidence intervals using the binomial proportion confidence interval (normal approximation)
  z_value <- qnorm(1 - (1 - confperc) / 2)  # z-value for confidence interval
  calibration_data <- calibration_data %>%
    mutate(
      lower_ci = mean_obs - z_value * sqrt((mean_obs * (1 - mean_obs)) / n),
      upper_ci = mean_obs + z_value * sqrt((mean_obs * (1 - mean_obs)) / n)
    )
  
  # Plotting the calibration curve with ggplot2
  plot <- ggplot(calibration_data, aes(x = mean_pred, y = mean_obs)) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.02) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +  # Perfect calibration line
    labs(
      x = "Predicted Probability",
      y = "Observed Proportion",
      title = "Calibration Curve with 95% Confidence Intervals"
    ) +
    xlim(0.15, 1.1)+
    ylim(-.3, 1.3)+
    theme_minimal()
  
  return(list(calibration_data = calibration_data, plot = plot))
}

#MODELING
library(caret)

set.seed(21)
data_split = createDataPartition(modeling_data$kick_distance, p = 0.75, list = FALSE)
model_train = modeling_data[data_split, ]
model_test = modeling_data[-data_split, ]


#linear model
set.seed(21)
linear_model = train(
  form = binary_result ~ poly(kick_distance, 3)+elevation+sqrt(windspeed)+temperature+precipitation+turf,
  data = model_train,
  trControl = trainControl(method = "cv", number = 5),
  preProc = c("nzv", "range"),
  method = "glm",
  family = "binomial"
)

print(linear_model)

confusion_matrix_glm <- caret::confusionMatrix(predict(linear_model, newdata = model_test), model_test$binary_result)
print(confusion_matrix_glm)

accuracy <- confusion_matrix_glm$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

#roc <- pROC::roc(predict(linear_model, newdata = model_test), model_test$binary_result)
#auc <- pROC::auc(roc)
#plot(roc, main = paste("AUC:", auc))


f1_score <- MLmetrics::F1_Score(predict(linear_model, newdata = model_test), model_test$binary_result)
print(paste("F1 Score:", f1_score))

GLMImp <- varImp(linear_model)
plot(GLMImp)

linear_model_df <- data.frame(Model = "Linear",
                                 Training_Accuracy = mean(linear_model$results$Accuracy) ,
                                 Test_Accuracy = accuracy,
                                 #AUC = auc,
                                 F1_Score = f1_score)


create_calibration_plot(model = linear_model, data = modeling_data, actual_col = binary_result)



#pen linear model
set.seed(21)
penalized_linear_model = train(
  form = binary_result ~ poly(kick_distance, 3)+elevation+sqrt(windspeed)+temperature+precipitation+turf,
  data = model_train,
  trControl = trainControl(method = "cv", number = 5),
  preProc = c("nzv", "range"),
  method = "glmnet",
  family = "binomial"
)

print(penalized_linear_model)

confusion_matrix_glm <- caret::confusionMatrix(predict(penalized_linear_model, newdata = model_test), model_test$binary_result)
print(confusion_matrix_glm)

accuracy <- confusion_matrix_glm$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

#roc <- pROC::roc(predict(penalized_linear_model, newdata = model_test), model_test$binary_result)
#auc <- pROC::auc(roc)
#plot(roc, main = paste("AUC:", auc))


f1_score <- MLmetrics::F1_Score(predict(penalized_linear_model, newdata = model_test), model_test$binary_result)
print(paste("F1 Score:", f1_score))

GLMImp <- varImp(penalized_linear_model)
plot(GLMImp)

pen_linear_model_df <- data.frame(Model = "Regularized",
                              Training_Accuracy = mean(penalized_linear_model$results$Accuracy) ,
                              Test_Accuracy = accuracy,
                              #AUC = auc,
                              F1_Score = f1_score)


create_calibration_plot(model = penalized_linear_model, data = modeling_data, actual_col = binary_result)

##RANDOM FOREST
set.seed(21)
random_forest = train(
  form = binary_result ~ kick_distance+elevation+windgusts+temperature+precipitation+turf,
  data = model_train,
  trControl = trainControl(method = "cv", number = 5),
  preProc = c("nzv", "range"),
  method = "rf",
  family = "binomial"
)

print(random_forest)

confusion_matrix_glm <- caret::confusionMatrix(predict(random_forest, newdata = model_test), model_test$binary_result)
print(confusion_matrix_glm)

accuracy <- confusion_matrix_glm$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

#roc <- pROC::roc(predict(penalized_linear_model, newdata = model_test), model_test$binary_result)
#auc <- pROC::auc(roc)
#plot(roc, main = paste("AUC:", auc))


f1_score <- MLmetrics::F1_Score(predict(random_forest, newdata = model_test), model_test$binary_result)
print(paste("F1 Score:", f1_score))

GLMImp <- varImp(random_forest)
plot(GLMImp)

rf_model_df <- data.frame(Model = "Random Forest",
                              Training_Accuracy = mean(random_forest$results$Accuracy) ,
                              Test_Accuracy = accuracy,
                              #AUC = auc,
                              F1_Score = f1_score)


create_calibration_plot(model = random_forest, data = modeling_data, actual_col = binary_result)


### XGB
set.seed(21)
XGB = train(
  form = binary_result ~ kick_distance+elevation+windspeed+temperature+precipitation+turf,
  data = model_train,
  trControl = trainControl(method = "cv", number = 5),
  preProc = c("nzv", "range"),
  method = "xgbTree",
  family = "binomial"
)

print(XGB)

confusion_matrix_glm <- caret::confusionMatrix(predict(XGB, newdata = model_test), model_test$binary_result)
print(confusion_matrix_glm)

accuracy <- confusion_matrix_glm$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

#roc <- pROC::roc(predict(penalized_linear_model, newdata = model_test), model_test$binary_result)
#auc <- pROC::auc(roc)
#plot(roc, main = paste("AUC:", auc))


f1_score <- MLmetrics::F1_Score(predict(XGB, newdata = model_test), model_test$binary_result)
print(paste("F1 Score:", f1_score))

GLMImp <- varImp(XGB)
plot(GLMImp)

xgb_model_df <- data.frame(Model = "XGB",
                              Training_Accuracy = mean(XGB$results$Accuracy) ,
                              Test_Accuracy = accuracy,
                              #AUC = auc,
                              F1_Score = f1_score)


create_calibration_plot(model = XGB, data = modeling_data, actual_col = binary_result)

### Neural Net
set.seed(21)
net = train(
  form = binary_result ~ kick_distance+elevation+windgusts+temperature+precipitation+turf,
  data = model_train,
  trControl = trainControl(method = "cv", number = 5),
  preProc = c("nzv", "range"),
  method = "nnet",
  family = "binomial"
)

print(net)

confusion_matrix_glm <- caret::confusionMatrix(predict(net, newdata = model_test), model_test$binary_result)
print(confusion_matrix_glm)

accuracy <- confusion_matrix_glm$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

#roc <- pROC::roc(predict(penalized_linear_model, newdata = model_test), model_test$binary_result)
#auc <- pROC::auc(roc)
#plot(roc, main = paste("AUC:", auc))


f1_score <- MLmetrics::F1_Score(predict(net, newdata = model_test), model_test$binary_result)
print(paste("F1 Score:", f1_score))

GLMImp <- varImp(net)
plot(GLMImp)

nn_model_df <- data.frame(Model = "Neural Net",
                              Training_Accuracy = mean(net$results$Accuracy) ,
                              Test_Accuracy = accuracy,
                              #AUC = auc,
                              F1_Score = f1_score)


create_calibration_plot(model = net, data = modeling_data, actual_col = binary_result)


model_df_list = list(linear_model_df, 
                     pen_linear_model_df, rf_model_df, xgb_model_df, nn_model_df)


model_summary_df <- do.call(rbind,model_df_list)
rownames(model_summary_df) <- NULL
print(model_summary_df)

#saveRDS(XGB, "NFL_Field_Goal_Model_XGB.rds")



modeling_data$pred <-  predict(XGB, modeling_data, type = "prob")
modeling_data$fg_percent_chance <- modeling_data$pred$`1`

merge_data <- modeling_data |> select(c(game_id, play_id,fg_percent_chance))
fg_results <- merge_data |>
  left_join(fg_plays, by = c("play_id", "game_id"))

fg_results$time <- as.character(fg_results$time)
fg_results$time <- as.character(fg_results$time)

fg_results <- fg_results %>%
  separate(time, into = c("minute", "second"), sep = ":") %>%
  mutate(Minute = as.integer(minute), Second = as.integer(second))


results_df <- fg_results |>
  select(c(posteam, week, Minute, season, home_team, game_date, qtr, down, yardline_100, kicker_player_name, kicker_player_id, fg_percent_chance, binary_result)) |>
  rename(STUD = fg_percent_chance)|>
  mutate(STUDplus = binary_result-STUD) |>
  filter(season == 2022)

kicker_results <- results_df |>
  group_by(kicker_player_name, kicker_player_id)|>
  summarise(totalSTUDplus = sum(STUDplus),
            meanSTUDplus = mean(STUDplus),
            avgSTUD = mean(STUD), 
            kicks = n(),
            FG_rate = round(mean(binary_result)*100),
            normMeanSTUDplus = totalSTUDplus/(avgSTUD*kicks), 
            meanSTUDpoints = meanSTUDplus*3,
            totalSTUDpoint = round(totalSTUDplus*3, 2)) |>
  filter(kicks >10)
  
meannormmean = mean(kicker_results$normMeanSTUDplus)
sdnormmean = sd(kicker_results$normMeanSTUDplus)

kicker_results$znormMeanSTUDplus = (kicker_results$normMeanSTUDplus - meannormmean) / sdnormmean
kicker_results$percentile = round(pnorm(kicker_results$znormMeanSTUDplus)*100, 1)

table_df <- kicker_results |>
  select(c(kicker_player_id, percentile, totalSTUDpoint, FG_rate, kicks )) |>
  arrange(-percentile) |>
  rename(Kicker = kicker_player_id, 
         "STUD+ Grade" = percentile, 
         "STUD+ Points" = totalSTUDpoint, 
         "FG%" = FG_rate, 
         "Attempts" = kicks) 



library(gt)


table_df |> head(5) |>
  gt() |> 
  cols_width(Kicker ~ px(100),
    
    everything() ~ px(100)
  ) |>
  cols_align(
    align = "center",
    columns = c("STUD+ Grade",
                "STUD+ Points", 
                "FG%", 
                "Attempts")
  ) |>
  cols_align(
    align = "left",
    columns = Kicker
  ) |>
  nflplotR::gt_nfl_headshots(columns = gt::ends_with("er"), height = 50) |> 
  # align the complete table left
  tab_options(
    table.align = "left"
  ) |>

  tab_header(title = md("**2022 NFL STUD+ Kicker Rankings**")) 


table_df |> tail(5) |>
  gt() |> 
  cols_width(Kicker ~ px(100),
             
             everything() ~ px(100)
  ) |>
  cols_align(
    align = "center",
    columns = c("STUD+ Grade",
                "STUD+ Points", 
                "FG%", 
                "Attempts")
  ) |>
  cols_align(
    align = "left",
    columns = Kicker
  ) |>
  nflplotR::gt_nfl_headshots(columns = gt::ends_with("er"), height = 50) |> 
  # align the complete table left
  tab_options(
    table.align = "left"
  ) |>
  
  tab_header(title = md("**2022 NFL STUD+ Kicker Rankings**")) 



############THIS DOENT WORK RIGHT NOW :()