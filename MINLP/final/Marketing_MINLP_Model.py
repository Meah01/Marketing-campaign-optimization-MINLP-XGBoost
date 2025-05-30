"""
Marketing Campaign Optimization Model using Mixed-Integer Nonlinear Programming (MINLP)

This module provides optimization functions for marketing campaigns to maximize either
impressions (awareness) or clicks (conversion) within budget constraints while accounting
for diminishing returns.

The model uses Pyomo with the Bonmin solver to solve MINLP problems.
"""

import os
import time
import numpy as np
import logging
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths for Bonmin solver
DEFAULT_BONMIN_PATH = "C:\\Users\\USER\\Desktop\\Optimization stuff\\coin-module.mswin64\\ampl.mswin64\\bonmin.exe"

# Optimal parameters for diminishing returns functions (from calibration)
AWARENESS_PARAMS = {
    'power': -0.3,
    'scale': 80.0,
    'saturation': 8.0
}

CONVERSION_PARAMS = {
    'power': -0.3,
    'scale': 100.0,
    'saturation': 10.0
}

# Platform performance coefficients
PLATFORM_COEFFICIENTS = {
    'facebook': {'impressions': 1.05, 'clicks': 1.03},
    'google': {'impressions': 1.00, 'clicks': 1.00},
    'dv360': {'impressions': 0.95, 'clicks': 0.98}
}

# Media cost polynomial coefficients (from data analysis)
MEDIA_COST_TO_IMPRESSIONS = {
    'poly_a': -0.4346,  # coefficient for x²
    'poly_b': 84.5877,  # coefficient for x
    'poly_c': 0.0000  # constant term
}

MEDIA_COST_TO_CLICKS = {
    'poly_a': -0.0197,  # coefficient for x²
    'poly_b': 3.0045,  # coefficient for x
    'poly_c': 0.0000  # constant term
}

# Campaign constraints
MIN_DURATION = 1
MAX_DURATION = 103
MIN_MEDIA_COST = 0.01


def diminishing_returns_effect(duration, power_param, scale_param, saturation_param):
    """
    Calculate diminishing returns effect with adjustable parameters.

    Parameters:
    -----------
    duration : float or int
        Campaign duration
    power_param : float
        Power parameter (controls how quickly returns diminish)
    scale_param : float
        Scale parameter (controls overall magnitude)
    saturation_param : float
        Saturation parameter (controls where the curve levels off)

    Returns:
    --------
    float
        Effect multiplier
    """
    # Formula: scale * (1 - exp(-duration/saturation)) * duration^power
    # When power is negative, returns diminish
    # Saturation parameter controls when the curve starts to level off
    return scale_param * (1 - np.exp(-duration / saturation_param)) * (duration ** power_param)


def optimize_campaign(campaign_settings):
    """
    Optimize a marketing campaign using Pyomo with Bonmin solver,
    accounting for diminishing returns over time.

    Parameters:
    -----------
    campaign_settings : dict
        Dictionary containing campaign settings:
        - campaign_goal: 'awareness' or 'conversion'
        - budget: float, total budget for the campaign
        - platform: str, one of 'facebook', 'google', 'dv360'
        - bonmin_path: str, path to Bonmin solver (optional)
        - dr_params: dict, custom diminishing returns parameters (optional)

    Returns:
    --------
    dict
        Dictionary containing optimization results and performance metrics
    """
    # Extract settings with defaults
    campaign_goal = campaign_settings.get('campaign_goal')
    budget = campaign_settings.get('budget')
    platform = campaign_settings.get('platform')
    bonmin_path = campaign_settings.get('bonmin_path', DEFAULT_BONMIN_PATH)
    dr_params = campaign_settings.get('dr_params', None)

    start_time = time.time()

    # Log start of optimization
    logger.info(f"Starting optimization for {campaign_goal} campaign on {platform} with budget ${budget}")

    # Validate inputs
    if campaign_goal not in ['awareness', 'conversion']:
        raise ValueError("campaign_goal must be either 'awareness' or 'conversion'")

    if platform not in ['facebook', 'google', 'dv360']:
        raise ValueError("platform must be one of 'facebook', 'google', or 'dv360'")

    if budget <= 0:
        raise ValueError("budget must be greater than 0")

    # Define relationship coefficients based on campaign goal
    if campaign_goal == 'awareness':
        # Impressions vs. Media Cost coefficients
        media_cost_poly_a = MEDIA_COST_TO_IMPRESSIONS['poly_a']
        media_cost_poly_b = MEDIA_COST_TO_IMPRESSIONS['poly_b']
        media_cost_poly_c = MEDIA_COST_TO_IMPRESSIONS['poly_c']

        # Use default diminishing returns parameters if not provided
        if dr_params is None:
            dr_params = AWARENESS_PARAMS
    else:
        # Clicks vs. Media Cost coefficients
        media_cost_poly_a = MEDIA_COST_TO_CLICKS['poly_a']
        media_cost_poly_b = MEDIA_COST_TO_CLICKS['poly_b']
        media_cost_poly_c = MEDIA_COST_TO_CLICKS['poly_c']

        # Use default diminishing returns parameters if not provided
        if dr_params is None:
            dr_params = CONVERSION_PARAMS

    # Get the appropriate platform coefficient
    platform_coef = PLATFORM_COEFFICIENTS[platform]['impressions'] if campaign_goal == 'awareness' else \
    PLATFORM_COEFFICIENTS[platform]['clicks']

    # Store DR parameters for use in post-processing
    power_param = dr_params['power']
    scale_param = dr_params['scale']
    saturation_param = dr_params['saturation']

    # Create a concrete model
    model = pyo.ConcreteModel()

    # Decision variables
    model.duration = pyo.Var(domain=pyo.Integers, bounds=(MIN_DURATION, MAX_DURATION))
    model.media_cost = pyo.Var(domain=pyo.NonNegativeReals, bounds=(MIN_MEDIA_COST, None))

    # Budget constraint
    def budget_rule(model):
        return model.media_cost * model.duration <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # For awareness campaigns
    if campaign_goal == 'awareness':
        # Use a simpler objective that captures the essence of diminishing returns
        def awareness_objective(model):
            # Media cost component
            media_cost_effect = media_cost_poly_a * model.media_cost ** 2 + media_cost_poly_b * model.media_cost

            # Simplified duration effect with diminishing returns
            duration_effect = model.duration ** (1 + power_param)

            return media_cost_effect * duration_effect * platform_coef

        model.objective = pyo.Objective(rule=awareness_objective, sense=pyo.maximize)

    # For conversion campaigns
    else:
        # Similar objective but with different coefficients
        def conversion_objective(model):
            # Media cost component
            media_cost_effect = media_cost_poly_a * model.media_cost ** 2 + media_cost_poly_b * model.media_cost

            # Simplified duration effect with diminishing returns
            duration_effect = model.duration ** (1 + power_param)

            return media_cost_effect * duration_effect * platform_coef

        model.objective = pyo.Objective(rule=conversion_objective, sense=pyo.maximize)

    # Check if Bonmin exists at the specified path
    if not os.path.exists(bonmin_path):
        logger.error(f"Bonmin solver not found at {bonmin_path}")
        return None

    # Create the solver
    solver = SolverFactory('bonmin', executable=bonmin_path)

    # Set solver options
    solver.options['bonmin.algorithm'] = 'B-BB'
    solver.options['bonmin.time_limit'] = 600  # 10 minutes

    # Solve the model
    try:
        results = solver.solve(model, tee=False)  # Set tee=True to see solver output

        # Check solver status
        solver_status = results.solver.status
        termination_condition = results.solver.termination_condition

        if solver_status == SolverStatus.ok and termination_condition == TerminationCondition.optimal:
            # Extract solution
            duration_value = int(pyo.value(model.duration))
            media_cost_value = pyo.value(model.media_cost)
            total_cost = duration_value * media_cost_value

            # Calculate performance metrics
            solve_time = time.time() - start_time

            # Performance metrics
            metrics = {
                'solver_time': solve_time,
                'solver_status': str(solver_status),
                'termination_condition': str(termination_condition)
            }

            # Now that we have the solution, we can calculate using the full diminishing returns function
            # Media cost component
            media_cost_effect = media_cost_poly_a * media_cost_value ** 2 + media_cost_poly_b * media_cost_value + media_cost_poly_c

            # Duration component without diminishing returns (linear)
            linear_duration_effect = duration_value

            # Duration component with diminishing returns using the function
            dr_effect = diminishing_returns_effect(
                duration_value,
                power_param,
                scale_param,
                saturation_param
            )

            # Calculate performance with and without diminishing returns
            if campaign_goal == 'awareness':
                # Calculate impressions
                linear_impressions = media_cost_effect * linear_duration_effect * platform_coef
                dr_impressions = media_cost_effect * dr_effect * platform_coef

                # Calculate error as percentage difference
                prediction_error = abs(
                    linear_impressions - dr_impressions) / linear_impressions * 100 if linear_impressions > 0 else 0

                metrics['linear_impressions'] = linear_impressions
                metrics['dr_impressions'] = dr_impressions
                metrics['prediction_error'] = prediction_error

                # Create results dictionary
                optimization_results = {
                    'campaign_goal': campaign_goal,
                    'platform': platform,
                    'budget': budget,
                    'duration': duration_value,
                    'media_cost': media_cost_value,
                    'total_cost': total_cost,
                    'impressions': dr_impressions,
                    'cpm': (total_cost / dr_impressions) * 1000 if dr_impressions > 0 else None,
                    'performance_metrics': metrics,
                    'dr_params': dr_params
                }
            else:
                # Calculate clicks
                linear_clicks = media_cost_effect * linear_duration_effect * platform_coef
                dr_clicks = media_cost_effect * dr_effect * platform_coef

                # Calculate error as percentage difference
                prediction_error = abs(linear_clicks - dr_clicks) / linear_clicks * 100 if linear_clicks > 0 else 0

                metrics['linear_clicks'] = linear_clicks
                metrics['dr_clicks'] = dr_clicks
                metrics['prediction_error'] = prediction_error

                # Create results dictionary
                optimization_results = {
                    'campaign_goal': campaign_goal,
                    'platform': platform,
                    'budget': budget,
                    'duration': duration_value,
                    'media_cost': media_cost_value,
                    'total_cost': total_cost,
                    'clicks': dr_clicks,
                    'cpc': total_cost / dr_clicks if dr_clicks > 0 else None,
                    'performance_metrics': metrics,
                    'dr_params': dr_params
                }

            logger.info(f"Optimization completed in {solve_time:.2f} seconds")
            logger.info(f"Optimal duration: {duration_value} days, Daily media cost: ${media_cost_value:.2f}")

            return optimization_results
        else:
            logger.error(
                f"Optimization failed with status {solver_status} and termination condition {termination_condition}")
            return None

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return None


def get_campaign_metrics(optimization_results):
    """
    Extract the essential campaign metrics from optimization results.
    This function is designed to be called from other files to access key metrics.

    Parameters:
    -----------
    optimization_results : dict
        Dictionary containing optimization results

    Returns:
    --------
    dict
        Dictionary containing only the essential campaign metrics
    """
    if not optimization_results:
        return None

    # Extract campaign goal
    campaign_goal = optimization_results.get('campaign_goal')

    # Common metrics
    metrics = {
        'campaign_goal': campaign_goal,
        'platform': optimization_results.get('platform'),
        'budget': optimization_results.get('budget'),
        'duration': optimization_results.get('duration'),
        'media_cost': optimization_results.get('media_cost'),
        'total_cost': optimization_results.get('total_cost')
    }

    # Goal-specific metrics
    if campaign_goal == 'awareness':
        metrics['impressions'] = optimization_results.get('impressions')
        metrics['cpm'] = optimization_results.get('cpm')
    else:
        metrics['clicks'] = optimization_results.get('clicks')
        metrics['cpc'] = optimization_results.get('cpc')

    return metrics


def compare_platforms(campaign_settings):
    """
    Compare optimization results across different platforms.

    Parameters:
    -----------
    campaign_settings : dict
        Dictionary containing campaign settings:
        - campaign_goal: 'awareness' or 'conversion'
        - budget: float, total budget for the campaign
        - bonmin_path: str, path to Bonmin solver (optional)
        - dr_params: dict, custom diminishing returns parameters (optional)

    Returns:
    --------
    list
        List of dictionaries containing optimization results for each platform
    """
    platforms = ['facebook', 'google', 'dv360']
    results = []

    for platform in platforms:
        # Create a copy of settings with this platform
        platform_settings = campaign_settings.copy()
        platform_settings['platform'] = platform

        # Run optimization for this platform
        logger.info(f"Optimizing for platform: {platform}")
        platform_result = optimize_campaign(platform_settings)

        if platform_result:
            results.append(platform_result)

    if not results:
        logger.warning("No results to compare. Check the solver configuration and try again.")
        return None

    return results


def sensitivity_analysis(campaign_settings, budget_range):
    """
    Perform sensitivity analysis by varying the budget.

    Parameters:
    -----------
    campaign_settings : dict
        Dictionary containing campaign settings:
        - campaign_goal: 'awareness' or 'conversion'
        - platform: str, one of 'facebook', 'google', 'dv360'
        - bonmin_path: str, path to Bonmin solver (optional)
        - dr_params: dict, custom diminishing returns parameters (optional)
    budget_range : list
        List of budget values to analyze

    Returns:
    --------
    list
        List of dictionaries containing optimization results for each budget level
    """
    results = []

    for budget in budget_range:
        # Create a copy of settings with this budget
        budget_settings = campaign_settings.copy()
        budget_settings['budget'] = budget

        # Run optimization for this budget
        logger.info(f"Optimizing for budget: ${budget}")
        budget_result = optimize_campaign(budget_settings)

        if budget_result:
            results.append(budget_result)

    if not results:
        logger.warning("No results to analyze. Check the solver configuration and try again.")
        return None

    return results


# Example usage (testing purposes)
if __name__ == "__main__":
    # Test the optimization with sample settings
    test_settings = {
        'campaign_goal': 'awareness',
        'budget': 1000,
        'platform': 'facebook'
    }

    # Run the optimization
    results = optimize_campaign(test_settings)

    # Print the results
    if results:
        print("\nOptimization Results:")
        for key, value in results.items():
            if key not in ['performance_metrics', 'dr_params']:
                print(f"{key}: {value}")

        # Extract essential metrics
        metrics = get_campaign_metrics(results)
        print("\nEssential Campaign Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")