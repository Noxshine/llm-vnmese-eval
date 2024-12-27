from Levenshtein import distance
from torchmetrics.text import CharErrorRate, WordErrorRate


def calculate_cer(pred, target):
    """
    Calculate the Character Error Rate (CER) between two strings.

    Args:
        pred (str): The ground truth string.
        targets (str): The predicted string.

    Returns:
        float: The CER value.
    """
    cer = CharErrorRate()
    return cer(pred, target).item()

def calculate_wer(pred, targets):
    wer = WordErrorRate()
    return wer(pred, targets).item()


def calculate_ced(str1, str2):
    '''
    Calculate Levenshtein_distance

    :param str1:
    :param str2:
    :return:
    '''
    return distance(str1, str2)


def calculate_wed(predicted, reference):
    '''
    Calculate word edit distance

    :param reference:
    :param predicted:
    :return:
    '''
    # Split the reference and predicted texts into words
    reference_words = reference.split()
    predicted_words = predicted.split()

    len_ref = len(reference_words)
    len_pred = len(predicted_words)

    # Create a matrix to store distances
    matrix = [[0] * (len_pred + 1) for _ in range(len_ref + 1)]

    # Initialize the first row and column
    for i in range(len_ref + 1):
        matrix[i][0] = i
    for j in range(len_pred + 1):
        matrix[0][j] = j

    # Fill the matrix with the WED distance values
    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            cost = 0 if reference_words[i - 1] == predicted_words[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,  # Deletion
                               matrix[i][j - 1] + 1,  # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

    # The WED is the number of errors divided by the number of words in the reference
    total_errors = matrix[len_ref][len_pred]
    # wed_value = total_errors / len_ref

    return total_errors

def calculate_ppl():
    pass