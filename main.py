import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os


def create_dir_to_save_plot(path_):
    if not os.path.isdir(f'{path_}/'):
        os.makedirs(f'{path_}/')


# Функція для побудови графіків
def plot_membership_function(x, y_values, labels, title):
    fig, axes = plt.subplots(1, len(y_values), figsize=(18, 6))
    for ax, y, label in zip(axes, y_values, labels):
        ax.plot(x, y, linewidth=1.5, label=label)
        ax.set_title(f"{title} - {label}")
        ax.legend()
    plt.tight_layout()
    path_to_save_plot_ = 'membership functions'
    create_dir_to_save_plot(path_to_save_plot_)
    plt.savefig(f"{path_to_save_plot_}/{title}.png")
    # plt.show()


def calculate_diagnosis(params):
    temperature_, cough_, throat_, nose_ = params
    diagnosing.input['temperature'] = temperature_
    diagnosing.input['cough'] = cough_
    diagnosing.input['throat'] = throat_
    diagnosing.input['nose'] = nose_
    diagnosing.compute()
    return diagnosing.output['diagnosis']


def interpret_diagnosis_to_text(fuzzy_result_diagnosis, diag_membership_values):
    membership_not = fuzz.interp_membership(diag_membership_values.universe, diag_membership_values['not'].mf,
                                            fuzzy_result_diagnosis)
    membership_low = fuzz.interp_membership(diag_membership_values.universe, diag_membership_values['low'].mf,
                                            fuzzy_result_diagnosis)
    membership_middle = fuzz.interp_membership(diag_membership_values.universe, diag_membership_values['middle'].mf,
                                               fuzzy_result_diagnosis)
    membership_high = fuzz.interp_membership(diag_membership_values.universe, diag_membership_values['high'].mf,
                                             fuzzy_result_diagnosis)
    membership_very_high = fuzz.interp_membership(diag_membership_values.universe,
                                                  diag_membership_values['very_high'].mf, fuzzy_result_diagnosis)
    memberships = {"Not": membership_not,
                   "Low": membership_low,
                   "Middle": membership_middle,
                   "High": membership_high,
                   "Very High": membership_very_high
                   }
    return max(memberships, key=memberships.get)


# Універсум дискурсу для змінних
x_temp = np.arange(29.5, 46.5, 0.1)
x_cough = np.arange(0.0, 10, 0.1)
x_throat = np.arange(0.0, 10, 0.1)
x_nose = np.arange(0.0, 10, 0.1)
y_diagnosis = np.arange(0.0, 10, 0.1)

# Задання змінних
temp = ctrl.Antecedent(x_temp, 'temperature')
cough = ctrl.Antecedent(x_cough, 'cough')
throat = ctrl.Antecedent(x_throat, 'throat')
nose = ctrl.Antecedent(x_nose, 'nose')
diagnosis = ctrl.Consequent(y_diagnosis, 'diagnosis')

# Задання функцій належності
temp['low'] = fuzz.trapmf(temp.universe, [29.5, 29.5, 35.5, 36.6])
temp['mild'] = fuzz.trimf(temp.universe, [35.5, 36.6, 37.2])
temp['high'] = fuzz.trapmf(temp.universe, [36.6, 37.2, 46.5, 46.5])

cough['none'] = fuzz.trapmf(cough.universe, [0, 0, 4, 5])
cough['mild'] = fuzz.trimf(cough.universe, [4, 5, 6])
cough['severe'] = fuzz.trapmf(cough.universe, [5, 6, 10, 10])

throat['none'] = fuzz.trapmf(throat.universe, [0, 0, 4, 5])
throat['mild'] = fuzz.trimf(throat.universe, [4, 5, 6])
throat['severe'] = fuzz.trapmf(throat.universe, [5, 6, 10, 10])

nose['none'] = fuzz.trapmf(nose.universe, [0, 0, 4, 5])
nose['mild'] = fuzz.trimf(nose.universe, [4, 5, 6])
nose['severe'] = fuzz.trapmf(nose.universe, [5, 6, 10, 10])

diagnosis['not'] = fuzz.trimf(diagnosis.universe, [0, 0, 1])
diagnosis['low'] = fuzz.trapmf(diagnosis.universe, [0, 1, 2, 3])
diagnosis['middle'] = fuzz.trapmf(diagnosis.universe, [3, 4, 5, 6])
diagnosis['high'] = fuzz.trapmf(diagnosis.universe, [6, 7, 8, 9])
diagnosis['very_high'] = fuzz.trimf(diagnosis.universe, [9, 10, 10])

# Побудова графіків для кожної змінної
plot_membership_function(x_temp, [temp['low'].mf, temp['mild'].mf, temp['high'].mf], ['Low', 'Medium', 'High'],
                         'Temperature')
plot_membership_function(x_cough, [cough['none'].mf, cough['mild'].mf, cough['severe'].mf], ['None', 'Mild', 'Severe'],
                         'Cough')
plot_membership_function(x_throat, [throat['none'].mf, throat['mild'].mf, throat['severe'].mf],
                         ['None', 'Mild', 'Severe'], 'Throat Pain')
plot_membership_function(x_nose, [nose['none'].mf, nose['mild'].mf, nose['severe'].mf], ['None', 'Mild', 'Severe'],
                         'Nose Congestion')
plot_membership_function(y_diagnosis,
                         [diagnosis['not'].mf, diagnosis['low'].mf, diagnosis['middle'].mf, diagnosis['high'].mf,
                          diagnosis['very_high'].mf], ['Not', 'Low', 'Middle', 'High', 'Very High'], 'Diagnosis')

# Задання правил
# Задання правил
rules = [
    ctrl.Rule(temp['mild'] & cough['none'] & throat['none'] & nose['none'], diagnosis['not']),
    ctrl.Rule(temp['high'] & cough['mild'] & throat['mild'] & nose['mild'], diagnosis['high']),

    ctrl.Rule(temp['high'], diagnosis['high']),
    ctrl.Rule(cough['severe'], diagnosis['high']),
    ctrl.Rule(throat['severe'], diagnosis['high']),
    ctrl.Rule(nose['severe'], diagnosis['high']),

    ctrl.Rule(temp['high'] & cough['severe'], diagnosis['very_high']),
    ctrl.Rule(cough['severe'] & throat['severe'], diagnosis['very_high']),
    ctrl.Rule(throat['severe'] & nose['severe'], diagnosis['very_high']),

    ctrl.Rule(temp['low'] & cough['mild'], diagnosis['middle']),
    # ctrl.Rule(cough['mild'] & throat['mild'], diagnosis['middle']),
    # ctrl.Rule(cough['mild'] & nose['mild'], diagnosis['middle']),
    # ctrl.Rule(throat['mild'] & nose['mild'], diagnosis['middle']),

    ctrl.Rule(nose['mild'], diagnosis['low']),
    ctrl.Rule(throat['mild'], diagnosis['low']),
    ctrl.Rule(cough['mild'], diagnosis['low'])
]

# Створення системи контролю
diagnosis_ctrl = ctrl.ControlSystem(rules)
diagnosing = ctrl.ControlSystemSimulation(diagnosis_ctrl)


# Тестування різних комбінацій вхідних значень
# combinations = [
#     [36.6, 0.0, 0.0, 0.0, 5.0, 2.0],
#     [38.0, 4.0, 3.0, 1.0, 3.0, 4.0],
#     [36.5, 1.0, 1.0, 2.0, 1.0, 1.0],
#     [37.5, 2.5, 2.5, 1.5, 2.5, 3.0],
#     [40.0, 5.0, 5.0, 5.0, 5.0, 5.0]]

# # age_, temperature_, cough_, throat_, nose_, headache_, fatigue_
# diagnosis_result_arr_ = []
# for comb in combinations:
#     diagnosis_result = calculate_diagnosis(comb)
#     diagnosis_result_arr_.append(diagnosis_result)
#     print(f"Вхідні значення: {comb} -> {diagnosis_result}")
#
# for diagnosis_result in diagnosis_result_arr_:
#     diagnosis_text = interpret_diagnosis_to_text(diagnosis_result, diagnosis)
#     print(f"Diagnosis: {diagnosis_text} Risk of ARVI ({(diagnosis_result * 10):.3f}%)")

def get_user_input():
    temperature_ = float(input("Enter temperature: "))
    cough_ = float(input("Enter cough level (intuitively, from 0 to 10): "))
    throat_ = float(input("Enter sore throat level (intuitively, from 0 to 10): "))
    nose_ = float(input("Enter nasal congestion level (intuitively, from 0 to 10): "))
    return [temperature_, cough_, throat_, nose_]


user_input_params = get_user_input()

diagnosis_result = calculate_diagnosis(user_input_params)
diagnosis_text = interpret_diagnosis_to_text(diagnosis_result, diagnosis)
print(f"You inputted following data: {dict(zip(['temperature', 'cough', 'sore throat', 'nasal congestion'], user_input_params))}")
print(f"Diagnosis: {diagnosis_text} Risk of ARVI ({(diagnosis_result * 10):.3f}%)")
