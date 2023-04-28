def calculate_average(lis):
    return round(sum(lis)/len(lis), 5)

top_1 = {
    'Accuracy': 0.6988,
    'Error Rate': 0.3012,
    'Sensitivity': 0.62857,
    'Specificity': 0.75,
    'Precision': 0.64706,
    'Recall': 0.62857,
    'F1': 0.63768
}

top_2 = {
    'Accuracy': 0.72289,
    'Error Rate': 0.27711,
    'Sensitivity': 0.65714,
    'Specificity': 0.77083,
    'Precision': 0.67647,
    'Recall': 0.65714,
    'F1': 0.66667,
}

top_3 = {
    'Accuracy': 0.6988,
    'Error Rate': 0.3012,
    'Sensitivity': 0.54286,
    'Specificity': 0.8125,
    'Precision': 0.67857,
    'Recall': 0.54286,
    'F1': 0.60317
}

top_4 = {
    'Accuracy': 0.73494,
    'Error Rate': 0.26506,
    'Sensitivity': 0.45714,
    'Specificity': 0.9375,
    'Precision': 0.84211,
    'Recall': 0.45714,
    'F1': 0.59259,
}

top_5 = {
    'Accuracy': 0.71084,
    'Error Rate': 0.28916,
    'Sensitivity': 0.62857,
    'Specificity': 0.77083,
    'Precision': 0.66667,
    'Recall': 0.62857,
    'F1': 0.64706,
}

se_1 = {
    'Accuracy': 0.74699,
    'Error Rate': 0.25301,
    'Sensitivity': 0.54286,
    'Specificity': 0.89583,
    'Precision': 0.79167,
    'Recall': 0.54286,
    'F1': 0.64407,
}

se_2 = {
    'Accuracy': 0.74699,
    'Error Rate': 0.25301,
    'Sensitivity': 0.57143,
    'Specificity': 0.875,
    'Precision': 0.76923,
    'Recall': 0.57143,
    'F1': 0.65574,
}

se_3 = {
    'Accuracy': 0.72289,
    'Error Rate': 0.27711,
    'Sensitivity': 0.62857,
    'Specificity': 0.79167,
    'Precision': 0.6875,
    'Recall': 0.62857,
    'F1': 0.65672,
}

se_4 = {
    'Accuracy': 0.6747,
    'Error Rate': 0.3253,
    'Sensitivity': 0.6,
    'Specificity': 0.72917,
    'Precision': 0.61765,
    'Recall': 0.6,
    'F1': 0.6087,
}

se_5 = {
    'Accuracy': 0.77108,
    'Error Rate': 0.22892,
    'Sensitivity': 0.71429,
    'Specificity': 0.8125,
    'Precision': 0.73529,
    'Recall': 0.71429,
    'F1': 0.72464,
}


def calc_avgs():
    average_accuracy_top = calculate_average(
        [top_1['Accuracy'], top_2['Accuracy'], top_3['Accuracy'], top_4['Accuracy'], top_5['Accuracy']]
    )
    average_specificity_top = calculate_average(
        [top_1['Specificity'], top_2['Specificity'], top_3['Specificity'], top_4['Specificity'], top_5['Specificity']]
    )
    average_precision_top = calculate_average(
        [top_1['Precision'], top_2['Precision'], top_3['Precision'], top_4['Precision'], top_5['Precision']]
    )
    average_recall_top = calculate_average(
        [top_1['Recall'], top_2['Recall'], top_3['Recall'], top_4['Recall'], top_5['Recall']]
    )
    average_f1_top = calculate_average(
        [top_1['F1'], top_2['F1'], top_3['F1'], top_4['F1'], top_5['F1']]
    )

    print(f'\t{average_f1_top}\t{average_accuracy_top}\t{average_precision_top}\t{average_recall_top}\t{average_specificity_top}')

    average_accuracy_se = calculate_average(
        [se_1['Accuracy'], se_2['Accuracy'], se_3['Accuracy'], se_4['Accuracy'], se_5['Accuracy']]
    )
    average_specificity_se = calculate_average(
        [se_1['Specificity'], se_2['Specificity'], se_3['Specificity'], se_4['Specificity'], se_5['Specificity']]
    )
    average_precision_se = calculate_average(
        [se_1['Precision'], se_2['Precision'], se_3['Precision'], se_4['Precision'], se_5['Precision']]
    )
    average_recall_se = calculate_average(
        [se_1['Recall'], se_2['Recall'], se_3['Recall'], se_4['Recall'], se_5['Recall']]
    )
    average_f1_se = calculate_average(
        [se_1['F1'], se_2['F1'], se_3['F1'], se_4['F1'], se_5['F1']]
    )
    
    print(f'\t{average_f1_se}\t{average_accuracy_se}\t{average_precision_se}\t{average_recall_se}\t{average_specificity_se}')

calc_avgs()