process AIUPRED {
    tag { meta.id }
    label 'aiupred'

    publishDir path: { "${params.outdir}/${meta.id}" }, mode: 'copy'

    input:
    tuple val(meta), path(fasta)

    output:
    tuple val(meta), path('*.aiupred.tsv'), emit: tsv

    script:
    def prefix = meta.id ?: fasta.baseName
    def bindFlag = params.aiupred.binding ? '-b' : ''
    def linkFlag = params.aiupred.linker ? '-l' : ''
    def redoxFlag = params.aiupred.redox ? '-r' : ''
    def gpuFlag = params.aiupred.force_cpu ? '' : "-g ${params.aiupred.gpu}"
    def cpuFlag = params.aiupred.force_cpu ? '--force-cpu' : ''
    """
    aiupred -i "${fasta}" -o "${prefix}.aiupred.tsv" ${bindFlag} ${linkFlag} ${redoxFlag} ${gpuFlag} ${cpuFlag}
    """
}
