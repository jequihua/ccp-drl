# load packages
library(restoptr)
library(terra)

################################################################################
#' Is a binary raster?
is_binary_raster <- function(x) {
  assertthat::assert_that(inherits(x, "SpatRaster"))
  terra::global(x, function(x) {
    x <- x[is.finite(x)]
    all(x == 0 | x == 1)
  })[[1]]
}
################################################################################
#' preprocess_input (aggregate to coarser resolution)
preprocess_input_fixed <- function (habitat, habitat_threshold = 1, aggregation_factor = 1){
  assertthat::assert_that(inherits(habitat, "SpatRaster"), 
                          habitat_threshold >= 0 && habitat_threshold <= 1, aggregation_factor >= 
                            1)
  assertthat::assert_that(terra::hasValues(habitat), terra::nlyr(habitat) == 
                            1)
  assertthat::assert_that(is_binary_raster(habitat), msg = "argument to \"habitat\" must have binary values")
  
  all_ones <- habitat >= 0
  
  cell_area <- terra::aggregate(all_ones, fact = aggregation_factor, 
                                  fun = "sum", na.rm = TRUE)*1.0
  
  down_sum <- terra::aggregate(habitat, fact = aggregation_factor, 
                                 fun = "sum", na.rm = TRUE)*1.0
  
  if (aggregation_factor==1 && habitat_threshold <1){
    warning(paste("The habitat threshold parameter has no effect when the", 
                  "aggregation factor is 1"))
    habitat_threshold <- 1
  }
  
  downsampled_habitat <- ((down_sum/cell_area) >= habitat_threshold) * 1.0
  
  restorable_habitat <- cell_area - down_sum
  
  downsampled_habitat <- downsampled_habitat - 0.1 + 0.1 
  restorable_habitat <- restorable_habitat - 0.1 + 0.1 
  cell_area <- cell_area - 0.1 + 0.1
  
  print(is.int(downsampled_habitat))
  print(is.int(restorable_habitat))
  print(is.int(cell_area))
  
  print(is.bool(downsampled_habitat))
  print(is.bool(restorable_habitat))
  print(is.bool(cell_area))

  return(list(existing_habitat = downsampled_habitat, restorable_habitat = restorable_habitat, 
              cell_area = cell_area))
}
################################################################################
#' Define restopt porblem.
restopt_problem_fixed <- function (existing_habitat, habitat_threshold = 1, aggregation_factor = 1){
  assertthat::assert_that(inherits(existing_habitat, "SpatRaster"))
  if (aggregation_factor == 1 && habitat_threshold < 1){
    warning(paste("The habitat threshold parameter was automatically set to 1,", 
                  "as the aggregation factor is 1"))
    habitat_threshold <- 1
  }
  preprocessed <- preprocess_input_fixed(habitat = existing_habitat, 
                                   habitat_threshold = habitat_threshold, aggregation_factor = aggregation_factor)
  
  habitat_down <- preprocessed$existing_habitat
  restorable_down <- preprocessed$restorable_habitat
  cell_area <- preprocessed$cell_area
  levels(habitat_down) <- data.frame(id = c(0, 1), label = c(paste("< ", 
                                                                   habitat_threshold * 100, "% habitat"), 
                                                             paste("≥ ", 
                                                                   habitat_threshold * 100, "% habitat")))
  names(habitat_down) <- "Existing habitat (aggregated)"
  plot(habitat_down, main = "habitat_down")
  names(restorable_down) <- "Restorable habitat (aggregated)"
  plot(restorable_down, main = "restorable_down")
  names(cell_area) <- "Cell area (aggregated)"
  plot(cell_area, main = "Cell area")
  set_no_objective(structure(list(data = list(original_habitat = existing_habitat, 
                                              existing_habitat = habitat_down, 
                                              restorable_habitat = restorable_down, 
                                              aggregation_factor = aggregation_factor, 
                                              cell_area = cell_area, 
                                              habitat_threshold = habitat_threshold, 
                                              locked_out = round(restorable_down < 0)),
                                                            constraints = list(), 
                                                            objective = NULL, 
                                                            settings = list(precision = 4L, 
                                                            time_limit = 0L, 
                                                            nb_solutions = 1L,
                                                            optimality_gap = 0, 
                                                            solution_name_prefix = "Solution ")), 
                             class = "RestoptProblem"))
}

# load habitat data
forest_2021 <- rast(system.file(
  "extdata", "case_study", "forest_2021.tif",
  package = "restoptr"
))

# visualize habitat data
plot(forest_2021)
forest_2021

# choose a small section of the raster
forest_2021_s <- forest_2021[200:230, 200:230, drop = FALSE]
plot(forest_2021_s)


# produce aggregated data
forest_2021_aggr <- preprocess_input(forest_2021_s,
                                     aggregation_factor = 3,
                                     habitat_threshold = 0.75)$existing_habitat


# create problems with constraints
p1 <- restopt_problem_fixed(
                           existing_habitat = forest_2021_aggr,
                           aggregation_factor = 1,
                           habitat_threshold = 1
                           )%>%
  
set_max_iic_objective()%>%
  
  add_restorable_constraint(
    min_restore = 6,
    max_restore = 6,
    unit = "cells"
)

solu <- solve(p1)

# 10 minutes
# (solving time = 589.08 s)
plot(solu)
