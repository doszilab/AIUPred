#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { AIUPRED } from './modules/local/aiupred'

workflow {
    ch_input = Channel.fromPath(params.input, checkIfExists: true)
        .map { fasta -> tuple([id: fasta.baseName], fasta) }

    AIUPRED(ch_input)
}
