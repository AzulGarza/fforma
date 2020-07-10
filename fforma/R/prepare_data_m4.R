
# libraries ---------------------------------------------------------------

pacman::p_load(tidyverse, data.table, furrr)


# data --------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
dir_ <- args[1]

print("Loading data...")
for(file in c("meta_M4", "submission_M4")){
  this_file <- str_glue("{dir_}/raw/decompressed_data/M4metaresults/data/{file}.rda")
  load(this_file)
}


# cleaning ----------------------------------------------------------------

clean_series <- function(list_ts, kind = "errors"){
  if(kind == "ff"){
    df <- list_ts[[kind]]
    df <- t(df)
    df <- as_tibble(df)
    df <- mutate(df, ds = 1:nrow(df))
    df <- select(df, ds, everything())

  } else if (kind=="freq") {
    ts <- list_ts[["x"]]
    df <- as_tibble(list(freq = frequency(ts), class_ts = class(ts)))

  } else if (kind=="xx") {
    ts <- as.vector(list_ts[[kind]])
    df <- enframe(ts, 'ds')
    df <- rename(df, y = value)

  } else {
    df <- as_tibble(as.list(list_ts[[kind]]))
  }

  df <- mutate(df, unique_id = list_ts$st)
  df <- select(df, unique_id, everything())

  return(df)
}


# multiprocessing task ----------------------------------------------------

plan(multiprocess)


# checking dirs -----------------------------------------------------------

dir_processed = str_glue("{dir_}/processed_data/")
if(!dir.exists(dir_processed)){
  dir.create(dir_processed)
}


# Saving train phase ------------------------------------------------------

print("Saving train data...")
for(kind in c("features", "ff", "xx")){
  print(kind)

  file_kind <- str_glue("{dir_}/processed_data/train-{kind}.csv")

  if(!file.exists(file_kind)){
    meta_M4 %>%
      future_map_dfr(clean_series, kind = kind, .progress = TRUE) %>%
      fwrite(file_kind)
  }
}

rm(meta_M4)

# Saving prediction phase -------------------------------------------------

print("Saving test data...")
for(kind in c("features", "ff")){
  print(kind)

  file_kind <- str_glue("{dir_}/processed_data/test-{kind}.csv")

  if(!file.exists(file_kind)){
    submission_M4 %>%
      future_map_dfr(clean_series, kind = kind, .progress = TRUE) %>%
      fwrite(file_kind)
  }
}
