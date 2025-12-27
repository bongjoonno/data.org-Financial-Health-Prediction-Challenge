from imports import pd
from constants import STRING_TO_NUM_TARGET

def clean_data(df, one_hot_encode_categoricals: bool = True, using_test_set: bool = False):
  df = df.copy()
  df['attitude_stable_business_environment'] = df['attitude_stable_business_environment'].fillna('Don’t know or N/A')

  df['attitude_worried_shutdown'] = df['attitude_worried_shutdown'].fillna('Don’t know or N/A')

  df['compliance_income_tax'] = df['compliance_income_tax'].fillna('Don’t know')

  df['perception_insurance_doesnt_cover_losses'] = df['perception_insurance_doesnt_cover_losses'].fillna("Don't know")

  df['perception_cannot_afford_insurance'] = df['perception_cannot_afford_insurance'].fillna("Don't know")

  df['personal_income'] = df['personal_income'].fillna(df['personal_income'].median())

  df['business_expenses'] = df['business_expenses'].fillna(df['business_expenses'].median())

  df['business_turnover'] = df['business_turnover'].fillna(df['business_turnover'].median())

  df['business_age_years'] = df['business_age_years'].fillna(df['business_age_years'].median())

  df['motor_vehicle_insurance'] = df['motor_vehicle_insurance'].fillna("Don't know")

  df['has_mobile_money'] = df['has_mobile_money'].fillna("Don't know")

  df['current_problem_cash_flow'] = df['current_problem_cash_flow'].fillna("Don't know")
  df['current_problem_cash_flow'] = df['current_problem_cash_flow'].str.replace('0', 'No')

  df['has_cellphone'] = df['has_cellphone'].fillna("Don't know")

  df['owner_sex'] = df['owner_sex'].fillna("Don't know")

  df['offers_credit_to_customers'] = df['offers_credit_to_customers'].fillna("Don't know")

  df['attitude_satisfied_with_achievement'] = df['attitude_satisfied_with_achievement'].fillna("Don't know")
  df['attitude_satisfied_with_achievement'] = df['attitude_satisfied_with_achievement'].str.replace('Don’t know or N/A', "Don't know")

  df['has_credit_card'] = df['has_credit_card'].fillna("Don't know")
  df['has_credit_card'] = df['has_credit_card'].str.replace('Used to have but don’t have now', "Used to have but don't have now")

  df['keeps_financial_records'] = df['keeps_financial_records'].fillna("Don't know")

  df['perception_insurance_companies_dont_insure_businesses_like_yours'] = df['perception_insurance_companies_dont_insure_businesses_like_yours'].fillna("Don't know")
  df['perception_insurance_companies_dont_insure_businesses_like_yours'] = df['perception_insurance_companies_dont_insure_businesses_like_yours'].str.replace('Don?t know / doesn?t apply', "Don't know")
  df['perception_insurance_companies_dont_insure_businesses_like_yours'] = df['perception_insurance_companies_dont_insure_businesses_like_yours'].str.replace("Don't Know", "Don't know")

  df['perception_insurance_important'] = df['perception_insurance_important'].fillna("Don't know")
  df['perception_insurance_important'] = df['perception_insurance_important'].str.replace(' Do not know / N\u200e/A', "Don't know")
  df['perception_insurance_important'] = df['perception_insurance_important'].str.replace("Don't Know", "Don't know")
  df['perception_insurance_important'] = df['perception_insurance_important'].str.replace('Don?t know / doesn?t apply', "Don't know")

  df['has_insurance'] = df['has_insurance'].fillna("Don't know")

  df['covid_essential_service'] = df['covid_essential_service'].fillna("Don't know")

  df['attitude_more_successful_next_year'] = df['attitude_more_successful_next_year'].fillna("Don't know")
  df['attitude_more_successful_next_year'] = df['attitude_more_successful_next_year'].str.replace('Don’t know or N/A', "Don't know")

  df['problem_sourcing_money'] = df['problem_sourcing_money'].fillna("Don't know")

  df['marketing_word_of_mouth'] = df['marketing_word_of_mouth'].fillna("Don't know")

  df['has_loan_account'] = df['has_loan_account'].fillna("Don't know")
  df['has_loan_account'] = df['has_loan_account'].str.replace('Used to have but don’t have now', "Used to have but don't have now")
  df['has_loan_account'] = df['has_loan_account'].str.replace('Don’t know (Do not show)', "Don't know")

  df['has_internet_banking'] = df['has_internet_banking'].fillna("Don't know")
  df['has_internet_banking'] = df['has_internet_banking'].str.replace('Used to have but don’t have now', "Used to have but don't have now")
  df['has_internet_banking'] = df['has_internet_banking'].str.replace('Don’t know (Do not show)', "Don't know")

  df['has_debit_card'] = df['has_debit_card'].fillna("Don't know")
  df['has_debit_card'] = df['has_debit_card'].str.replace('Used to have but don’t have now', "Used to have but don't have now")

  df['future_risk_theft_stock'] = df['future_risk_theft_stock'].fillna("Don't know")

  df['business_age_months'] = df['business_age_months'].fillna(df['business_age_months'].median())

  df['medical_insurance'] = df['medical_insurance'].fillna("Don't know")
  df['medical_insurance'] = df['medical_insurance'].str.replace('Used to have but don’t have now', "Used to have but don't have now")
  df['medical_insurance'] = df['medical_insurance'].str.replace('Don’t know (Do not show)', "Don't know")

  df['funeral_insurance'] = df['funeral_insurance'].fillna("Don't know")
  df['funeral_insurance'] = df['funeral_insurance'].str.replace('Used to have but don’t have now', "Used to have but don't have now")
  df['funeral_insurance'] = df['funeral_insurance'].str.replace('Don’t know (Do not show)', "Don't know")

  df['motivation_make_more_money'] = df['motivation_make_more_money'].fillna("Don't know")

  df['uses_friends_family_savings'] = df['uses_friends_family_savings'].fillna("Don't know")
  df['uses_friends_family_savings'] = df['uses_friends_family_savings'].str.replace('Used to have but don’t have now', "Used to have but don't have now")
  df['uses_friends_family_savings'] = df['uses_friends_family_savings'].str.replace('Don’t know (Do not show)', "Don't know")

  df['uses_informal_lender'] = df['uses_informal_lender'].fillna("Don't know")
  df['uses_informal_lender'] = df['uses_informal_lender'].str.replace('Used to have but don’t have now', "Used to have but don't have now")
  df['uses_informal_lender'] = df['uses_informal_lender'].str.replace('Don’t know (Do not show)', "Don't know")

  if one_hot_encode_categoricals:
    for column in df.columns:
      if df[column].dtype == object and column not in ['Target', 'ID']:
        df[column] = df[column].astype('category').cat.codes
  
  if not using_test_set:
    df['Target'] = df['Target'].map(STRING_TO_NUM_TARGET)

  return df