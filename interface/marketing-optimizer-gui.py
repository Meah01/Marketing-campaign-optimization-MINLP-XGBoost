import os
import sys
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# Path to MINLP model directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Add the MINLP model path
sys.path.insert(0, r"C:\Users\USER\Desktop\Thesis\MINLP\final")

# Try to import the MINLP model
try:
    from Marketing_MINLP_Model import optimize_campaign, get_campaign_metrics
    print("MINLP model imported successfully")
except ImportError as e:
    error_msg = (
        f"Could not import Marketing_MINLP_Model from:\n"
        f"C:\\Users\\USER\\Desktop\\Thesis\\MINLP\\final\n\n"
        f"Error: {str(e)}\n\n"
        f"Please ensure:\n"
        f"• The file Marketing_MINLP_Model.py exists in the specified directory\n"
        f"• The file contains optimize_campaign and get_campaign_metrics functions" #functions needed for optimisation
    )
    messagebox.showerror("Import Error", error_msg)
    sys.exit(1)

# Path to XGBoost models
XGBOOST_PATH = r"C:\Users\USER\Desktop\Thesis\XGBoost\final"


class MarketingOptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marketing Campaign Optimization")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Configure style
        style = ttk.Style()
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5")
        style.configure("TLabelframe", background="#f5f5f5")
        style.configure("TLabelframe.Label", background="#f5f5f5", font=("Arial", 10, "bold"))
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("Subheader.TLabel", font=("Arial", 10, "bold"))
        style.configure("Result.TLabel", font=("Arial", 10))
        style.configure("Run.TButton", font=("Arial", 10, "bold"))

        # Set up the main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Marketing Campaign Optimizer", style="Header.TLabel")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="w")

        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.grid(row=1, column=0, padx=(0, 10), pady=(0, 10), sticky="nsew")

        # Goal Choice
        ttk.Label(input_frame, text="Campaign Goal:").grid(row=0, column=0, sticky="w", pady=5)
        self.goal_var = tk.StringVar(value="Awareness")
        goal_combobox = ttk.Combobox(input_frame, textvariable=self.goal_var, state="readonly")
        goal_combobox['values'] = ('Awareness', 'Conversion')
        goal_combobox.grid(row=0, column=1, sticky="ew", pady=5)
        goal_combobox.bind("<<ComboboxSelected>>", self.on_goal_change)

        # Platform Choice
        ttk.Label(input_frame, text="Advertising Platform:").grid(row=1, column=0, sticky="w", pady=5)
        self.platform_var = tk.StringVar(value="Facebook Ads")
        platform_combobox = ttk.Combobox(input_frame, textvariable=self.platform_var, state="readonly")
        platform_combobox['values'] = ('Facebook Ads', 'Google Ads', 'DV360')
        platform_combobox.grid(row=1, column=1, sticky="ew", pady=5)

        # Budget Input
        ttk.Label(input_frame, text="Maximum Budget (USD):").grid(row=2, column=0, sticky="w", pady=5)
        self.budget_var = tk.StringVar(value="1000")
        budget_entry = ttk.Entry(input_frame, textvariable=self.budget_var)
        budget_entry.grid(row=2, column=1, sticky="ew", pady=5)

        # Goal Information
        self.goal_info_label = ttk.Label(input_frame, text="Goal: Optimize for Impressions",
                                         font=("Arial", 10, "italic"))
        self.goal_info_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Optimize Button
        optimize_button = ttk.Button(input_frame, text="Optimize Campaign", style="Run.TButton",
                                     command=self.optimize_campaign)
        optimize_button.grid(row=4, column=0, columnspan=2, pady=(15, 5), sticky="ew")

        # Output Frame
        output_frame = ttk.LabelFrame(main_frame, text="Optimization Results", padding="10")
        output_frame.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="nsew")

        # Optimization Results Section
        ttk.Label(output_frame, text="MINLP Optimization Results", style="Subheader.TLabel").grid(row=0, column=0,
                                                                                                  columnspan=2,
                                                                                                  sticky="w",
                                                                                                  pady=(0, 10))

        # Optimal Duration
        ttk.Label(output_frame, text="Optimal Duration (days):").grid(row=1, column=0, sticky="w", pady=3)
        self.duration_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.duration_var, style="Result.TLabel").grid(row=1, column=1, sticky="e",
                                                                                            pady=3)

        # Estimated Media Costs
        ttk.Label(output_frame, text="Estimated Media Costs (USD):").grid(row=2, column=0, sticky="w", pady=3)
        self.media_costs_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.media_costs_var, style="Result.TLabel").grid(row=2, column=1,
                                                                                               sticky="e", pady=3)

        # Estimated Goal Metric
        self.goal_metric_label = ttk.Label(output_frame, text="Estimated Impressions:")
        self.goal_metric_label.grid(row=3, column=0, sticky="w", pady=3)
        self.goal_metric_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.goal_metric_var, style="Result.TLabel").grid(row=3, column=1,
                                                                                               sticky="e", pady=3)

        # Separator
        ttk.Separator(output_frame, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

        # ML Prediction Results Section
        ttk.Label(output_frame, text="ML Predictions", style="Subheader.TLabel").grid(row=5, column=0, columnspan=2,
                                                                                      sticky="w", pady=(0, 10))

        # Estimated clicks/impressions (depending on goal)
        self.clicks_impressions_label = ttk.Label(output_frame, text="Estimated Clicks:")
        self.clicks_impressions_label.grid(row=6, column=0, sticky="w", pady=3)
        self.clicks_impressions_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.clicks_impressions_var, style="Result.TLabel").grid(row=6, column=1,
                                                                                                      sticky="e",
                                                                                                      pady=3)

        # CTR
        ttk.Label(output_frame, text="Estimated CTR:").grid(row=7, column=0, sticky="w", pady=3)
        self.ctr_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.ctr_var, style="Result.TLabel").grid(row=7, column=1, sticky="e",
                                                                                       pady=3)

        # CPC
        ttk.Label(output_frame, text="Estimated CPC (USD):").grid(row=8, column=0, sticky="w", pady=3)
        self.cpc_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.cpc_var, style="Result.TLabel").grid(row=8, column=1, sticky="e",
                                                                                       pady=3)

        # CPM
        ttk.Label(output_frame, text="Estimated CPM (USD):").grid(row=9, column=0, sticky="w", pady=3)
        self.cpm_var = tk.StringVar(value="-")
        ttk.Label(output_frame, textvariable=self.cpm_var, style="Result.TLabel").grid(row=9, column=1, sticky="e",
                                                                                       pady=3)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Load XGBoost models
        self.load_models()

    def load_models(self):
        """Loads the XGBoost prediction models"""
        try:
            # Load Awareness model (impressions ---> clicks)
            awareness_model_path = os.path.join(XGBOOST_PATH, "xgboost_awareness_nonlinear_model.pkl")
            with open(awareness_model_path, 'rb') as f:
                self.awareness_model_dict = pickle.load(f)

            # Load Conversion model (clicks ---> impressions)
            conversion_model_path = os.path.join(XGBOOST_PATH, "xgboost_conversion_nonlinear_model.pkl")
            with open(conversion_model_path, 'rb') as f:
                self.conversion_model_dict = pickle.load(f)

            self.status_var.set("Models loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error Loading Models", error_msg)

    def on_goal_change(self, event=None):
        """Updates the UI when the goal selection changes"""
        goal = self.goal_var.get()
        if goal == "Awareness":
            self.goal_info_label.config(text="Goal: Optimize for Impressions")
            self.goal_metric_label.config(text="Estimated Impressions:")
            self.clicks_impressions_label.config(text="Estimated Clicks:")
        else:  # Conversion
            self.goal_info_label.config(text="Goal: Optimize for Clicks")
            self.goal_metric_label.config(text="Estimated Clicks:")
            self.clicks_impressions_label.config(text="Estimated Impressions:")

    def optimize_campaign(self):
        """Running the optimization and showing results"""
        try:
            # Get input values
            goal = self.goal_var.get()
            platform = self.platform_var.get()

            try:
                budget = float(self.budget_var.get())
                if budget <= 0:
                    raise ValueError("Budget must be positive")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid positive number for budget.")
                return

            # Map platform to parameter expected by the model
            platform_mapping = {
                "Facebook Ads": "facebook",
                "Google Ads": "google",
                "DV360": "dv360"
            }
            platform_param = platform_mapping.get(platform, "facebook")

            # Update status
            self.status_var.set(f"Running optimization for {goal} campaign on {platform}...")
            self.root.update_idletasks()

            # Prepare settings for MINLP model
            campaign_settings = {
                'campaign_goal': goal.lower(),
                'budget': budget,
                'platform': platform_param
            }

            # Call the MINLP model
            optimization_results = optimize_campaign(campaign_settings)

            if not optimization_results:
                raise Exception("Optimization failed to produce valid results")

            # Extract results
            duration = optimization_results.get('duration', 0)
            media_cost = optimization_results.get('media_cost', 0)
            total_cost = optimization_results.get('total_cost', 0)

            # Get goal metric based on campaign type
            if goal == "Awareness":
                # For awareness campaigns, show impressions
                goal_metric = optimization_results.get('impressions', 0)
                self.goal_metric_var.set(f"{goal_metric:,.0f}")
            else:  # Conversion
                # For conversion campaigns, show clicks
                goal_metric = optimization_results.get('clicks', 0)
                self.goal_metric_var.set(f"{goal_metric:,.0f}")

            # Update UI with optimization results
            self.duration_var.set(f"{duration:.0f}")
            self.media_costs_var.set(f"${total_cost:.2f}")

            # Use ML model to predict additional metrics
            self.predict_additional_metrics(goal, platform_param, duration, media_cost, goal_metric)

            # Update status
            self.status_var.set(f"Optimization complete for {goal} campaign on {platform}")

        except Exception as e:
            error_msg = f"An error occurred during optimization: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Optimization Error", error_msg)

    def predict_additional_metrics(self, goal, platform, duration, media_cost, goal_metric):
        """XGBoost models to predict additional metrics"""
        try:
            # Create base feature vector for prediction - same for both models
            features = {
                'duration': duration,
                'media_cost_usd': media_cost,
                # Add squared and log-transformed features as required by the model
                'duration_squared': duration ** 2,
                'log_duration': np.log1p(duration),
                'log_media_cost': np.log1p(media_cost),
                'log_media_cost_squared': np.log1p(media_cost) ** 2,
                'log_media_cost_x_duration': np.log1p(media_cost) * np.log1p(duration),
                # Platform should be passed as a categorical feature
                'ext_service_name': platform,
                # Campaign segment might be needed - derive from goal
                'campaign_segment': 'awareness' if goal == "Awareness" else 'conversion'
            }

            total_cost = duration * media_cost

            # Debug information
            debug_info = {
                "Input Parameters": {
                    "goal": goal,
                    "platform": platform,
                    "duration": duration,
                    "media_cost": media_cost,
                    "goal_metric": goal_metric,
                    "total_cost": total_cost
                },
                "Features": features,
                "Selected Model Info": {},
                "Predictions": {},
                "Metrics": {}
            }

            # Process based on campaign goal
            if goal == "Awareness":
                # For Awareness campaigns
                impressions = goal_metric  # From MINLP

                # Create DataFrame for awareness model
                features_df = pd.DataFrame([features])
                debug_info["Features DataFrame"] = features_df.to_dict()

                # Use Awareness model to predict clicks
                try:
                    # Get model metadata
                    debug_info["Selected Model Info"]["model_name"] = "Awareness Model"
                    debug_info["Selected Model Info"]["model_type"] = type(self.awareness_model_dict["model"])
                    debug_info["Selected Model Info"]["use_log_target"] = self.awareness_model_dict.get(
                        'use_log_target', False)

                    # Make raw prediction
                    raw_prediction = self.awareness_model_dict['model'].predict(features_df)[0]
                    debug_info["Selected Model Info"]["raw_prediction"] = float(raw_prediction)

                    # Apply log transformation if needed
                    clicks = raw_prediction
                    if self.awareness_model_dict.get('use_log_target', False):
                        clicks = np.expm1(raw_prediction)
                        debug_info["Selected Model Info"]["expm1_applied"] = True

                    debug_info["Selected Model Info"]["final_prediction"] = float(clicks)

                except Exception as e:
                    error_msg = f"Error making prediction with Awareness Model: {str(e)}"
                    print(error_msg)
                    debug_info["Selected Model Info"]["error"] = error_msg
                    clicks = 0

                debug_info["Predictions"]["impressions_source"] = "MINLP"
                debug_info["Predictions"]["clicks_source"] = "Awareness Model"

                # Update UI with predictions
                self.clicks_impressions_var.set(f"{clicks:,.0f}")

            else:  # Conversion
                # For Conversion campaigns
                clicks = goal_metric  # From MINLP

                # Add clicks to features since Conversion Model expects it
                features['clicks'] = clicks

                # Create DataFrame for conversion model
                features_df = pd.DataFrame([features])
                debug_info["Features DataFrame"] = features_df.to_dict()

                # Use Conversion model to predict impressions
                try:
                    # Get model metadata
                    debug_info["Selected Model Info"]["model_name"] = "Conversion Model"
                    debug_info["Selected Model Info"]["model_type"] = type(self.conversion_model_dict["model"])
                    debug_info["Selected Model Info"]["use_log_target"] = self.conversion_model_dict.get(
                        'use_log_target', False)

                    # Make raw prediction
                    raw_prediction = self.conversion_model_dict['model'].predict(features_df)[0]
                    debug_info["Selected Model Info"]["raw_prediction"] = float(raw_prediction)

                    # Apply log transformation if needed
                    impressions = raw_prediction
                    if self.conversion_model_dict.get('use_log_target', False):
                        impressions = np.expm1(raw_prediction)
                        debug_info["Selected Model Info"]["expm1_applied"] = True

                    debug_info["Selected Model Info"]["final_prediction"] = float(impressions)

                except Exception as e:
                    error_msg = f"Error making prediction with Conversion Model: {str(e)}"
                    print(error_msg)
                    debug_info["Selected Model Info"]["error"] = error_msg
                    # Fallback to using assumed CTR if model prediction fails
                    assumed_ctr = 0.03  # 3%
                    impressions = clicks / assumed_ctr
                    debug_info["Predictions"]["fallback_applied"] = True
                    debug_info["Predictions"]["assumed_ctr"] = assumed_ctr
                    debug_info["Predictions"]["fallback_impressions"] = float(impressions)

                # Sanity check for impressions
                if impressions < clicks:
                    # Apply a reasonable assumption-based correction
                    # Assume a reasonable CTR of 3% for Facebook ads
                    assumed_ctr = 0.03  # 3%
                    corrected_impressions = clicks / assumed_ctr

                    debug_info["Predictions"]["correction_applied"] = True
                    debug_info["Predictions"]["original_impressions"] = float(impressions)
                    debug_info["Predictions"]["assumed_ctr"] = assumed_ctr
                    debug_info["Predictions"]["corrected_impressions"] = float(corrected_impressions)

                    impressions = corrected_impressions

                debug_info["Predictions"]["clicks_source"] = "MINLP"
                debug_info["Predictions"]["impressions_source"] = "Conversion Model (with correction if needed)"

                # Update UI with predictions
                self.clicks_impressions_var.set(f"{impressions:,.0f}")

            # Calculate CTR and cost metrics
            ctr = clicks / impressions if impressions > 0 else 0
            cpc = total_cost / clicks if clicks > 0 else 0
            cpm = (total_cost / impressions) * 1000 if impressions > 0 else 0

            # Update UI with metrics
            self.ctr_var.set(f"{ctr:.4f}")
            self.cpc_var.set(f"${cpc:.2f}")
            self.cpm_var.set(f"${cpm:.2f}")

            # Store final metrics in debug info
            debug_info["Metrics"] = {
                "impressions": float(impressions),
                "clicks": float(clicks),
                "ctr": float(ctr),
                "cpc": float(cpc),
                "cpm": float(cpm)
            }

            # Print debug information
            print("\n===== ML PREDICTION DEBUG INFO =====")
            import json
            print(json.dumps(debug_info, indent=2, default=str))
            print("===================================\n")

            # Additional sanity check warning
            if impressions < clicks:
                print(
                    f"WARNING: Predicted impressions ({impressions}) less than clicks ({clicks}). This is logically impossible.")
                print("Applying fallback calculation for impressions based on assumed CTR.")

        except Exception as e:
            error_msg = f"An error occurred during ML prediction: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Prediction Error", error_msg)
            # Print detailed error for debugging
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    # Create a mock optimizer for testing if the real one is not available
    if "optimize_campaign" not in globals():
        def optimize_campaign(campaign_settings):
            """Mock function for testing the UI"""
            import random
            import time

            # Simulate processing time
            time.sleep(1)

            if campaign_settings['campaign_goal'] == "awareness":
                impressions = campaign_settings['budget'] * random.randint(800, 1200)
                return {
                    'duration': random.randint(14, 60),
                    'media_cost': campaign_settings['budget'] / 30,
                    'total_cost': campaign_settings['budget'] * 0.9,
                    'impressions': impressions
                }
            else:  # conversion
                clicks = campaign_settings['budget'] * random.randint(40, 60) / 1000
                return {
                    'duration': random.randint(21, 45),
                    'media_cost': campaign_settings['budget'] / 30,
                    'total_cost': campaign_settings['budget'] * 0.85,
                    'clicks': clicks
                }

    root = tk.Tk()
    app = MarketingOptimizationApp(root)
    root.mainloop()