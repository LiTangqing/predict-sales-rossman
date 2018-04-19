library(lubridate)
library(ggplot2)
library(dplyr)
library(caret)
set.seed(4240)
library(mlr)
library(magrittr)
library(xgboost)

train.control <- trainControl(method = "cv", number = 5)

store.df <- read.csv("store.csv") %>%
  mutate(CompetitionOpenDate = as.POSIXct(
    ifelse(!is.na(CompetitionOpenSinceYear),
           paste(CompetitionOpenSinceYear,
                 CompetitionOpenSinceMonth,
                 "01", sep='-'),
           NA),
    format="%Y-%m-%d"))

(train.df <- read.csv("train.csv") %>%
    mutate(DayOfWeek = as.factor(DayOfWeek)) %>%
    mutate(Date = as.POSIXct(as.character(Date), format = "%Y-%m-%d")) %>%
    merge(store.df, by.x = "Store", by.y = "Store", all.x = T) %>% 
    mutate(HasCompetition = ifelse(is.na(CompetitionOpenDate), F, CompetitionOpenDate - Date < 0))  %>%
    mutate(CompetitionDistance = ifelse(is.na(CompetitionDistance), 0, CompetitionDistance * HasCompetition))  %>%
    filter(Open == 1) %>%
    mutate(LogSales = log(Sales + 1),
           LogCustomers = log(Customers + 1),
           LogCompetitionDistance = log(CompetitionDistance + 1)) %>%
    filter(Sales >= 1 & Customers >= 1) %>%
    mutate(InplementPromo2 = ifelse(Promo2 == 1 &
                                      year(Date) >= Promo2SinceYear & 
                                      week(Date) >= Promo2SinceWeek, 1, 0),
           PromoInterval = as.character(PromoInterval)) %>%
    mutate(OnPromo2 = ifelse(  InplementPromo2 == 1 & PromoInterval == "Feb,May,Aug,Nov" & 
                                 month(Date) %in% c(2, 5, 8, 11) |
                                 InplementPromo2 == 1 & PromoInterval == "Jan,Apr,Jul,Oct" & 
                                 month(Date) %in% c(1, 4, 7, 10) | 
                                 InplementPromo2 == 1 & PromoInterval == "Mar,Jun,Sept,Dec" & 
                                 month(Date) %in% c(3, 6, 9, 12), 1, 0)) %>%
    select(-c(InplementPromo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval, Open)) %>%
    mutate(MonthSinceComp = ifelse(HasCompetition == 1,
                                   interval(CompetitionOpenDate, Date) %/% months(1) ,0)) %>%
    mutate(LogMonthSinceComp = log(MonthSinceComp + 1)) %>%
    mutate(RvMonthSinceComp = ifelse(MonthSinceComp != 0, 1/MonthSinceComp, 0)) %>%
    mutate(Promo1Weekday = as.factor(
      ifelse(DayOfWeek == 6 | DayOfWeek == 7 | Promo == 0, 0, as.numeric(DayOfWeek))
    )) %>%
    mlr::createDummyFeatures(cols = "Promo1Weekday")
)

# construct sales and and customers from past quaterm half year and 2 yrs
# helper dataframe
(quarter.sales.df <- train.df %>%
    select(Date, Sales, Store) %>%
    mutate(Quarter = quarter(Date + months(3), with_year = T)) %>%
    group_by(Store, Quarter) %>%
    summarise(AvgSalesPrevQuarter = mean(Sales)))


train.2014 <- train.df %>%
  filter(Date > as.POSIXct("2014-01-02", format = "%Y-%m-%d")) %>%
  mutate(Quarter = quarter(Date, with_year = T)) %>%
  merge(y = quarter.sales.df, 
        by = c("Store", "Quarter")) %>%
  mutate(LogPrevQuarterSales = log(AvgSalesPrevQuarter)) 



 

train.2014 <- train.2014 %>%
  mutate_if(is.integer, as.numeric) %>%
  mutate(HasCompetition = as.numeric(HasCompetition))

train.2014 <- train.2014 %>%
  mlr::createDummyFeatures(cols = c("DayOfWeek", "StateHoliday", 
                                    "StoreType", "Assortment"))


# construct xgboost here
predictors <- c("Promo", "SchoolHoliday","LogCompetitionDistance", "HasCompetition",
                "OnPromo2", "RvMonthSinceComp", "Promo1Weekday.1",
                "LogPrevQuarterSales", "Promo1Weekday.2", 'Promo1Weekday.3',
                "Promo1Weekday.4", c(paste0(rep("DayOfWeek.", 7), 1:7)),
                "StateHoliday.0", "StateHoliday.a", "StateHoliday.b", 
                "StateHoliday.c", "StoreType.a","StoreType.b","StoreType.c",
                "StoreType.d","Assortment.a","Assortment.b","Assortment.c"
               )

#data <-  as.matrix(train.2014[, predictors])

dtrain <- xgb.DMatrix(data = as.matrix(train.2014[, predictors]), 
                      label = as.numeric(train.2014$LogSales))
# creat learner
lrn <-makeLearner("regr.xgboost", predict.type = "response")
lrn$par.vals<- list(objective = "reg:linear", eval_metric = "rmse",
                    nrounds = 100L, eta = .1)
# # set parameter space
params <- makeParamSet(makeDiscreteParam("booster", values = c("gbtree", "gblinear")),
                       makeIntegerParam("max_depth",lower = 3L,upper = 10L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

# # resampling strategy
rdesc <- makeResampleDesc("CV", iters=5L)
# # search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)
#
# # train task
full <- c(predictors, "LogSales")
traintask <- makeRegrTask(data = train.2014[, full], target = "LogSales")

library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                     measures = mse, par.set = params, control = ctrl, show.info = T)
mytune$y
# [Tune] Result: booster=gbtree; max_depth=10; min_child_weight=2.66;
# subsample=0.64; colsample_bytree=0.659 : mse.test.mean=0.0320304

# add LogPrevQuarterSales 
# RMSE       Rsquared  MAE      
# 0.2105937  0.749739  0.1534811

# final model
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

resample(lrn_tune, traintask, rdesc, measures = list(rmse))
# rmse = 0.1792084
 
(xgbFit <- xgboost(data = as.matrix(train.2014[, predictors]), 
                  label = as.numeric(train.2014$LogSales), nfold = 5, 
                  verbose = 1, objective = "reg:linear", nrounds = 2000,
                  eval_metric = "rmse", booster="gbtree", max_depth=10, min_child_weight=2.66,
                  subsample=0.64, colsample_bytree=0.659))

