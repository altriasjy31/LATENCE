/* Corrected ontology preparation for the supplied DeepGOZero JCEL libraries.
 * Runs with Java 11+ source-file mode. The upstream repository is not modified.
 * Class IDs are shared with the JCEL translator. Fresh classes have distinct,
 * parser-compatible ZC_0000000 names and are never treated as GO candidates.
 */
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import de.tudresden.inf.lat.jcel.owlapi.translator.*;
import de.tudresden.inf.lat.jcel.ontology.normalization.OntologyNormalizer;
import de.tudresden.inf.lat.jcel.ontology.axiom.extension.IntegerOntologyObjectFactory;
import de.tudresden.inf.lat.jcel.ontology.axiom.complex.ComplexIntegerAxiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.*;

class DeepGOZeroNormalizer {
    static String clean(String value) {
        return value.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
    }
    static String compact(IRI iri) {
        return iri.getShortForm().replace(':', '_');
    }
    static class Symbols {
        final TranslationRepository repository;
        final Map<Integer, String> classes = new TreeMap<>();
        final Map<Integer, String> roles = new TreeMap<>();
        final Set<String> occupied = new HashSet<>();
        final List<String> rows = new ArrayList<>();
        int nextClass = 0;
        int nextRole = 0;
        int specialBuiltinClasses = 0;

        Symbols(TranslationRepository repository, OWLOntology ontology) {
            this.repository = repository;
            // Reserve every original name, including names outside model NF1-4.
            for (OWLEntity e : ontology.getSignature()) occupied.add(compact(e.getIRI()));
            rows.add("kind\tinteger_id\tsymbol\toriginal_iri\tauxiliary\tspecial_builtin\n");
        }
        String fresh(boolean role) {
            String value;
            do {
                int index = role ? nextRole++ : nextClass++;
                if (index >= 10000000) throw new IllegalStateException("Auxiliary symbol space exhausted");
                value = String.format(java.util.Locale.ROOT, (role ? "ZR_" : "ZC_") + "%07d", index);
            } while (occupied.contains(value));
            occupied.add(value);
            return value;
        }
        String className(int id) {
            if (!classes.containsKey(id)) {
                Optional<OWLClass> original = repository.getOptOWLClass(id);
                String iri = original.isPresent() ? original.get().getIRI().toString() : "";
                String symbol = original.isPresent() ? compact(original.get().getIRI()) : "";
                // Native NF1 parser accepts class tokens of exactly length 10.
                // Preserve canonical GO IDs; map other IRI classes explicitly.
                if (!symbol.matches("GO_[0-9]{7}")) symbol = fresh(false);
                if (classes.containsValue(symbol)) {
                    throw new IllegalStateException("Distinct original class IRIs share a GO symbol: " + symbol);
                }
                classes.put(id, symbol);
                String special = original.isPresent() && original.get().isOWLThing() ? "owl:Thing" :
                    original.isPresent() && original.get().isOWLNothing() ? "owl:Nothing" : "none";
                if (!special.equals("none")) specialBuiltinClasses++;
                rows.add("class\t" + id + "\t" + symbol + "\t" + clean(iri) + "\t" + (!original.isPresent()) + "\t" + special + "\n");
            }
            return classes.get(id);
        }
        String roleName(int id) {
            if (!roles.containsKey(id)) {
                Optional<OWLObjectProperty> original = repository.getOptOWLObjectProperty(id);
                String iri = original.isPresent() ? original.get().getIRI().toString() : "";
                String symbol = original.isPresent() ? compact(original.get().getIRI()) : "";
                if (!symbol.matches("[A-Za-z0-9_:-]+") || roles.containsValue(symbol) ||
                    Arrays.asList("and", "some", "SubClassOf").contains(symbol)) symbol = fresh(true);
                roles.put(id, symbol);
                rows.add("role\t" + id + "\t" + symbol + "\t" + clean(iri) + "\t" + (!original.isPresent()) + "\tnone\n");
            }
            return roles.get(id);
        }
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("ontology-file output-directory");
        Path input = Paths.get(args[0]).toAbsolutePath();
        Path output = Paths.get(args[1]).toAbsolutePath();
        Files.createDirectories(output);
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        // Import IRIs resolve to a deliberately absent local file. Never query
        // a live URL behind an import declaration in a frozen experiment.
        File blockedImport = output.resolve("IMPORTS_BLOCKED_DO_NOT_CREATE.owl").toFile();
        if (blockedImport.exists()) throw new IllegalArgumentException("Reserved import-blocker file exists");
        manager.getIRIMappers().add(iri -> IRI.create(blockedImport));
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(input.toFile());
        if (!ontology.getImportsDeclarations().isEmpty()) {
            throw new IllegalArgumentException("Freeze and merge ontology imports before normalization");
        }
        // Use one shared factory for translating existing entities AND creating
        // normalization auxiliaries. A fresh second factory can reuse IDs.
        IntegerOntologyObjectFactory factory = new de.tudresden.inf.lat.jcel.ontology.axiom.extension.IntegerOntologyObjectFactoryImpl();
        Translator translator = new Translator(manager.getOWLDataFactory(), factory);
        translator.getTranslationRepository().addAxiomEntities(ontology);
        Set<ComplexIntegerAxiom> complex = new HashSet<>();
        List<String> translationFailures = new ArrayList<>();
        for (OWLAxiom axiom : ontology.getAxioms()) {
            try {
                Set<ComplexIntegerAxiom> translated = axiom.accept(translator.getAxiomTranslator());
                if (axiom.isLogicalAxiom() && translated.isEmpty()) {
                    translationFailures.add("empty_logical_translation\t" + clean(axiom.toString()));
                }
                complex.addAll(translated);
            } catch (RuntimeException error) {
                translationFailures.add(clean(error.getClass().getSimpleName() + ": " + error.getMessage()) +
                    "\t" + clean(axiom.toString()));
            }
        }
        Collections.sort(translationFailures);
        int failureCount = translationFailures.size();
        translationFailures.add(0, "reason\tinput_axiom");
        Files.write(output.resolve("untranslated_input_axioms.tsv"), translationFailures, StandardCharsets.UTF_8);
        if (failureCount != 0) throw new IllegalArgumentException(failureCount +
            " input axioms could not be translated; inspect untranslated_input_axioms.tsv. No partial normalization is certified.");
        Set<NormalizedIntegerAxiom> normalized = new OntologyNormalizer().normalize(complex, factory);
        Symbols symbols = new Symbols(translator.getTranslationRepository(), ontology);
        // Assign tokens in integer ID order, independently of Set iteration.
        TreeSet<Integer> classIDs = new TreeSet<>();
        TreeSet<Integer> roleIDs = new TreeSet<>();
        for (NormalizedIntegerAxiom ax : normalized) {
            if (ax instanceof GCI0Axiom) {
                GCI0Axiom a = (GCI0Axiom) ax; classIDs.add(a.getSubClass()); classIDs.add(a.getSuperClass());
            } else if (ax instanceof GCI1Axiom) {
                GCI1Axiom a = (GCI1Axiom) ax; classIDs.add(a.getLeftSubClass()); classIDs.add(a.getRightSubClass()); classIDs.add(a.getSuperClass());
            } else if (ax instanceof GCI2Axiom) {
                GCI2Axiom a = (GCI2Axiom) ax; classIDs.add(a.getSubClass()); classIDs.add(a.getClassInSuperClass()); roleIDs.add(a.getPropertyInSuperClass());
            } else if (ax instanceof GCI3Axiom) {
                GCI3Axiom a = (GCI3Axiom) ax; classIDs.add(a.getClassInSubClass()); classIDs.add(a.getSuperClass()); roleIDs.add(a.getPropertyInSubClass());
            }
        }
        for (int id : classIDs) symbols.className(id);
        for (int id : roleIDs) symbols.roleName(id);
        TreeSet<String> lines = new TreeSet<>();
        List<String> excluded = new ArrayList<>();
        Map<String, Integer> excludedCounts = new TreeMap<>();
        for (NormalizedIntegerAxiom ax : normalized) {
            String line;
            if (ax instanceof GCI0Axiom) {
                GCI0Axiom a = (GCI0Axiom) ax;
                line = symbols.className(a.getSubClass()) + " SubClassOf " + symbols.className(a.getSuperClass());
            } else if (ax instanceof GCI1Axiom) {
                GCI1Axiom a = (GCI1Axiom) ax;
                List<String> operands = new ArrayList<>(Arrays.asList(symbols.className(a.getLeftSubClass()), symbols.className(a.getRightSubClass())));
                Collections.sort(operands);
                line = operands.get(0) + " and " + operands.get(1) + " SubClassOf " + symbols.className(a.getSuperClass());
            } else if (ax instanceof GCI2Axiom) {
                GCI2Axiom a = (GCI2Axiom) ax;
                line = symbols.className(a.getSubClass()) + " SubClassOf " + symbols.roleName(a.getPropertyInSuperClass()) + " some " + symbols.className(a.getClassInSuperClass());
            } else if (ax instanceof GCI3Axiom) {
                GCI3Axiom a = (GCI3Axiom) ax;
                line = symbols.roleName(a.getPropertyInSubClass()) + " some " + symbols.className(a.getClassInSubClass()) + " SubClassOf " + symbols.className(a.getSuperClass());
            } else {
                String type = ax.getClass().getSimpleName();
                excludedCounts.put(type, excludedCounts.getOrDefault(type, 0) + 1);
                excluded.add(type + "\t" + clean(ax.toString()) + "\n");
                continue;
            }
            lines.add(line);
        }
        if (lines.isEmpty()) throw new IllegalStateException("No NF1-NF4 class axioms produced");
        Files.write(output.resolve("go.norm"), lines, StandardCharsets.UTF_8);
        Files.writeString(output.resolve("symbol_mapping.tsv"), String.join("", symbols.rows), StandardCharsets.UTF_8);
        Collections.sort(excluded);
        excluded.add(0, "type\tnormalized_integer_axiom\n");
        Files.writeString(output.resolve("excluded_normal_forms.tsv"), String.join("", excluded), StandardCharsets.UTF_8);
        List<String> stats = new ArrayList<>();
        stats.add("key\tvalue");
        stats.add("input_axioms\t" + ontology.getAxiomCount());
        stats.add("input_logical_axioms\t" + ontology.getLogicalAxiomCount());
        stats.add("translated_integer_axioms\t" + complex.size());
        stats.add("normalized_axioms\t" + normalized.size());
        stats.add("model_class_axioms\t" + lines.size());
        stats.add("excluded_nonmodel_axioms\t" + (excluded.size() - 1));
        stats.add("untranslated_input_axioms\t" + failureCount);
        stats.add("special_builtin_class_symbols\t" + symbols.specialBuiltinClasses);
        for (Map.Entry<String, Integer> entry : excludedCounts.entrySet()) stats.add("excluded_" + entry.getKey() + "\t" + entry.getValue());
        Files.write(output.resolve("normalizer_stats.tsv"), stats, StandardCharsets.UTF_8);
        System.out.println("Model NF1-NF4 axioms: " + lines.size() + "; separately reported nonmodel axioms: " + (excluded.size() - 1));
    }
}
