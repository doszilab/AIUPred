def multifasta_reader(file_handler):
    """
    (multi) FASTA reader function
    :return: Dictionary with header -> sequence mapping from the file
    """
    if type(file_handler) == str:
        file_handler = open(file_handler)
    sequence_dct = {}
    header = None
    for line in file_handler:
        if line.startswith('>'):
            header = line.strip()
            sequence_dct[header] = ''
        elif line.strip():
            sequence_dct[header] += line.strip()
    file_handler.close()
    return sequence_dct
