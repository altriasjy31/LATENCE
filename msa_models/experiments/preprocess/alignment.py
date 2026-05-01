# using alignment algorithm

from subprocess import Popen, PIPE, STDOUT
import os
# from functools import reduce

class HMMER(object):
    def __init__(self, binpath = ""):
        self.bin = binpath

    def jackhmmer(self, seqfile, dbfile, iterations, ofile, msafile, 
        evalue=None, incE=None, incdomE=None,
        tscore=None,incT=None,incdomT=None,
        idflag=False, idcutoff=0.95, threads=2):
        cmdlst = [
            os.path.join(self.bin, "jackhmmer"), 
            "-N {}".format(iterations),
            "-A {}".format(msafile),
            "--notextw",
            "--cpu {}".format(threads),
        ]

        if evalue is not None:
            cmdlst.append(f"-E {evalue}")
        
        if incE is not None:
            cmdlst.append(f"--incE {incE}")

        if incdomE is not None:
            cmdlst.append(f"--incdomE {incdomE}")

        if tscore is not None:
            cmdlst.append(f"-T {tscore}")
        
        if incT is not None:
            cmdlst.append(f"--incT {incT}")

        if incdomT is not None:
            cmdlst.append(f"--incdomT {incdomT}")

        if ofile is not None:
            cmdlst.append("-o {}".format(ofile))

        if idflag and (idcutoff is not None):
            cmdlst += [
                "--eclust",
                "-eid {}".format(idcutoff)
            ]
        
        
        cmdlst += [seqfile, dbfile]
        
        cmdstr = " ".join(cmdlst)

        status = Popen(cmdstr, stdout=PIPE,stderr=PIPE,shell=True)

        _, error = status.communicate()

        assert error.decode() == "",\
            "{}".format(error.decode())
    """
    msafile is the input
    hmmfile is the output
    """
    def hmmbuild(self, hmmfile, msafile, informat="stockholm", singlemx=True):
        cmdlst = ["hmmbuild"]

        cmdlst += [
            "--informat {}".format(informat),
            "--amino" # prevent from the msa only has one sequence
        ]

        if singlemx:
            cmdlst.append("--singlemx")

        cmdlst += [hmmfile, msafile]
        cmdstr = " ".join(cmdlst)
        
        status = Popen(cmdstr, stdout=PIPE, stderr=PIPE, shell=True)

        _, error = status.communicate()

        assert error.decode() == "", \
            "{}".format(error.decode())
    
    """
    """
    def hmmsearch(self, hmmfile, dbfile, msafile, ofile, 
                  notextwflag=True, evalue=None, incE=None, incdomE=None,
                  tscore=None,incT=None,incdomT=None,
                  threads=5):
        cmdlst = ["hmmsearch"]

        cmdlst += [
            "-A {}".format(msafile),
            "--cpu {}".format(threads)
        ]

        if evalue is not None:
            cmdlst.append(f"-E {evalue}")
        
        if incE is not None:
            cmdlst.append(f"--incE {incE}")

        if incdomE is not None:
            cmdlst.append(f"--incdomE {incdomE}")

        if tscore is not None:
            cmdlst.append(f"-T {tscore}")
        
        if incT is not None:
            cmdlst.append(f"--incT {incT}")

        if incdomT is not None:
            cmdlst.append(f"--incdomT {incdomT}")

        if ofile is not None:
            cmdlst.append("-o {}".format(ofile))
        
        if notextwflag:
            cmdlst.append("--notextw")

        cmdlst += [hmmfile, dbfile]

        cmdstr = " ".join(cmdlst)
        
        status = Popen(cmdstr, stdout=PIPE, stderr=PIPE, shell=True)

        _, error = status.communicate()
        
        # assert error.decode() == "", \
        #     "{}".format(error.decode())

    
    def reformat(self, hhlib):
        self.hhlib = hhlib
        def _perform(infmt, outfmt, ifile, ofile, textw=50000, gap="-"):
            cmdlst = [
                os.path.join(hhlib,"scripts","reformat.pl"),
                infmt,outfmt,
                ifile, ofile,
                f"-l {textw}"
            ]

            if gap is not None:
                cmdlst.append(f"-g {gap}")

            cmdstr = " ".join(cmdlst)

            status = Popen(cmdstr, stdout=PIPE, stderr=PIPE,shell=True)

            _, error = status.communicate()

            assert error.decode() == "",\
                "{}".format(error.decode())    
        return _perform    
 
    def reformat_mpi(self, hhlib, infmt, outfmt, ifile, ofile, textw=50000, gap="-"):
        self.hhlib = hhlib
        cmdlst = [
            os.path.join(hhlib,"scripts","reformat.pl"),
            infmt,outfmt,
            ifile, ofile,
            f"-l {textw}"
        ]

        if gap is not None:
            cmdlst.append(f"-g {gap}")
        cmdstr = " ".join(cmdlst)
        status = Popen(cmdstr, stdout=PIPE, stderr=PIPE,shell=True)
        _, error = status.communicate()
        # assert error.decode() == "",\
        #     "{}".format(error.decode())      


def jackhmmer_alignment(seqdir, dbfile, outdir, msadir, outfile=None, iterations=5, threads=2):
    # creating HMMER object
    using_hmmer = HMMER()
    name_format = "{}.fa"
    outname_format = "{}.out"
    msa_format = "{}.sto"
        
    def _single_process(accessid):
        # using the jackhmmer
        ifile = os.path.join(seqdir, name_format.format(accessid))
        msafile = os.path.join(msadir, msa_format.format(accessid))
        if outfile is None:
            ofile = os.path.join(outdir,outname_format.format(accessid))
        else:
            ofile = os.path.join(outdir,outname_format.format(outfile))
        using_hmmer.jackhmmer(
            ifile,dbfile,iterations,ofile,msafile,
            threads
        )
        
    return _single_process

def jackhmmer_align_mpi(accessid, seqdir, dbfile, outdir, msadir, outfile=None, iterations=5, 
    evalue=None, incE=None, incdomE=None,
    tscore=None,incT=None,incdomT=None,
    idflag=False, idcutoff=0.95, threads=2):
    using_hmmer = HMMER()
    name_format = "{}.fa"
    outname_format = "{}.out"
    msa_format = "{}.sto"

    ifile = os.path.join(seqdir, name_format.format(accessid))
    msafile = os.path.join(msadir, msa_format.format(accessid))
    if outfile is None:
        ofile = os.path.join(outdir,outname_format.format(accessid))
    else:
        ofile = os.path.join(outdir,outname_format.format(outfile))
    using_hmmer.jackhmmer(
        ifile,dbfile,iterations,ofile,msafile,
        evalue, incE, incdomE,
        tscore,incT,incdomT,
        idflag, idcutoff, threads
    )

def msa_reformat_mpi(accessid, indir, outdir, infmt, outfmt, hhlib, textw, gap="-"):
    using_hmmer = HMMER()
    in_format = "{}." + f"{infmt}"
    out_format = "{}." + f"{outfmt}"

    ifile = os.path.join(indir, in_format.format(accessid))
    ofile = os.path.join(outdir, out_format.format(accessid))

    using_hmmer.reformat_mpi(hhlib, infmt,outfmt,ifile,ofile,textw, gap)


fmt2surfix = {
    "stockholm": "sto",
    "a2m": "a2m",
    "a3m": "a3m"
}

def hmmbuild_mpi(accessid, indir, outdir, informat, singlemx=True):
    using_hmmer = HMMER()
    surfix = fmt2surfix[informat]
    ifilefmt = "{}." + f"{surfix}"
    hmmfilefmt = "{}.hmm"

    msafile = os.path.join(indir, ifilefmt.format(accessid))
    hmmfile = os.path.join(outdir, hmmfilefmt.format(accessid))

    using_hmmer.hmmbuild(hmmfile, msafile, informat, singlemx)

def hmmsearch_mpi(accessid, indir, dbfile, outdir, outfile, 
    notextwflag=False,evalue=None, incE=10, incdomE=10, threads=5):
    using_hmmer = HMMER()
    hmmfilefmt = "{}.hmm"
    msafilefmt = "{}.sto"

    hmmfile = os.path.join(indir, hmmfilefmt.format(accessid))
    msafile = os.path.join(outdir, msafilefmt.format(accessid))

    using_hmmer.hmmsearch(hmmfile,dbfile,msafile,outfile,
        notextwflag, evalue, incE, incdomE, threads)
