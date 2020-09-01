============
File formats
============

``deepnog`` uses standard file formats, as detailed below for
eggNOG 5 (1239, Firmicutes) data.

Protein sequences
=================
Protein sequences are expected in FASTA format.
Each entry must contain a unique record ID.
That is, a ``user_data.faa`` should look like this:

::

    >1000569.HMPREF1040_0002
    MMKHDDHVHQIRTEPIYAILGETFSRGRTNRQVAKALLGAGVRIIQYREKEKSWQEKYEE
    ARDICQWCNEYGATFIMNDSIDLAIACEAPAIHVGQDDAPVAWVRRLAQRDIVVGVSTHT
    IAEMKKAVRDGADYVGLGPMYQTTSKMDVHDIVADVDKAYALTLPIPVVTIGGIDLIHIR
    QLYTEGFRSFAMISALVGATDIVEQIGAFRQVLQEKIDEC
    >1000569.HMPREF1040_0003
    MATTVGDIVTYLQGIAPLYLKEEWDNPGLLLGNQGDPVSSVLVTLDVMEGTVDYAIAEGI
    SFIFSHHPLIMKGIKAIRTDSYDGRMYQKLLSHHIAVYAAHTNLDSATGGVNDVLAEHLQ
    LQHVRPFIPGVSESLYKIAIYVPKGYGDAIREVLGKHDAGHLGAYSYCSFSVAGQGRFKP
    LAGTHPFIGKRDVLETVEEERIETIVEGSRLGEVITAMLAVHPYEEPAYDIYPLYQQRTA
    LGLGRLGELATPLSSMAAVQWVKEALHLTHVSYAGPMDRQIQTIAVLGGSGAEFIATAKA
    AGATLYVTGDMKYHAAQEAIKQGILVVDAGHFGTEFPVIDRMKQNIEAENEKQGWHIQCV
    VDPTAMDMIQRL

Compression is allowed (``user_data.faa.gz``, or ``user_data.faa.xz``).
For typical usage of ``deepnog infer`` for protein orthologous group assignments
this is already sufficient.


Protein orthologous group labels
================================
Training new models with ``deepnog train``, or assessing model quality
with ``deepnog infer --test_labels`` require providing the orthologous group
labels.

File format is CSV (comma-separated values) with a preceding header line,
and three columns (index, sequence record ID, orthologous group ID).

::

    ,string_id,eggnog_id
    1543720,1121929.KB898683_gene1916,1V3NB
    351865,536232.CLM_3459,1TPCN
    [...]
    1570381,1000569.HMPREF1040_0002,1V3ZR
    744166,1000569.HMPREF1040_0003,1TQ27
    [...]
    426023,1423743.JCM14108_56,1TPGE

To construct some ``user_data.csv``:

* Copy (do not modify) the header line.
* Provide an index in the first column (e.g. 1..N; currently unused, but required).
* Provide the sequence ID (e.g. eggNOG/STRING ID) in column 2.
* Provide its corresponding group label in column 3.
* Sequence IDs in column 2 must match the IDs used in the ``user_data.faa``.


Assignment output
=================
Orthologous group assignments are output in tabular format (comma-separated).

* Column 1: Sequence ID
* Column 2: Assignment/Orthologous group
* Column 3: Assignment confidence in 0..1 (higher=better).

Example:

::

    sequence_id,prediction,confidence
    1000565.METUNv1_00038,COG0466,1.0
    1000565.METUNv1_00060,COG0500,0.20852506
    1000565.METUNv1_00091,COG0810,0.9999591
    1000565.METUNv1_00093,COG0659,1.0
    1000565.METUNv1_00103,COG5000,0.70716757
    1000565.METUNv1_00105,COG0346,0.9999982
    1000565.METUNv1_00106,COG3791,1.0
    1000565.METUNv1_00114,COG0239,1.0
    1000565.METUNv1_00115,COG1643,1.0
