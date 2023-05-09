const Field = require("@saltcorn/data/models/field");
const Table = require("@saltcorn/data/models/table");
const Form = require("@saltcorn/data/models/form");
const Workflow = require("@saltcorn/data/models/workflow");
const FieldRepeat = require("@saltcorn/data/models/fieldrepeat");
const { eval_expression } = require("@saltcorn/data/models/expression");
const fs = require("fs");
const fsp = fs.promises;
const {
  field_picker_fields,
  picked_fields_to_query,
  stateFieldsToWhere,
  stateFieldsToQuery,
  readState,
} = require("@saltcorn/data/plugin-helper");

const pythonBridge = require("python-bridge");

//https://stackoverflow.com/a/56095793/19839414
const util = require("util");
const exec = util.promisify(require("child_process").exec);

let python = pythonBridge({
  python: "python3",
});

const configuration_workflow = (req) =>
  new Workflow({
    steps: [
      {
        name: "Predictors",
        form: async (context) => {
          const table = await Table.findOne(
            context.table_id
              ? { id: context.table_id }
              : { name: context.exttable_name }
          );
          //console.log(context);
          const field_picker_repeat = await field_picker_fields({
            table,
            viewname: context.viewname,
            req,
            no_fieldviews: true,
          });

          const type_pick = field_picker_repeat.find((f) => f.name === "type");
          type_pick.attributes.options = type_pick.attributes.options.filter(
            ({ name }) =>
              ["Field", "JoinField", "Aggregation", "FormulaValue"].includes(
                name
              )
          );

          const use_field_picker_repeat = field_picker_repeat.filter(
            (f) =>
              !["state_field", "col_width", "col_width_units"].includes(f.name)
          );

          return new Form({
            fields: [
              new FieldRepeat({
                name: "columns",
                fancyMenuEditor: true,
                fields: use_field_picker_repeat,
              }),
            ],
          });
        },
      },
    ],
  });

const write_csv = async (rows, columns, fields, filename) => {
  return new Promise((resolve, reject) => {
    const colWriters = [];
    /*let idSupply = 0;
  const getId = () => {
    idSupply++;
    return idSupply;
  };*/
    columns.forEach((column) => {
      switch (column.type) {
        case "FormulaValue":
          colWriters.push({
            header: column.header_label,
            write: (row) => eval_expression(column.formula, row),
          });
          break;
        case "Field":
          let f = fields.find((fld) => fld.name === column.field_name);
          if (f.type.name === "FloatArray") {
            const dims = rows.map((r) => r[column.field_name].length);
            const maxDims = Math.max(...dims);
            for (let i = i < maxDims; i++; ) {
              colWriters.push({
                header: column.field_name + i,
                write: (row) => row[column.field_name][i],
              });
            }
          } else {
            colWriters.push({
              header: column.field_name,
              write: (row) => row[column.field_name],
            });
          }
          break;

        default:
          break;
      }
    });
    const outstream = fs.createWriteStream(filename);
    outstream.write(colWriters.map((cw) => cw.header).join(",") + "\n");
    rows.forEach((row) => {
      outstream.write(colWriters.map((cw) => cw.write(row)).join(",") + "\n");
    });
    outstream.end();
    //https://stackoverflow.com/a/39880990/19839414
    outstream.on("finish", () => {
      resolve();
    });
    outstream.on("error", reject);
  });
};

module.exports = {
  sc_plugin_api_version: 1,
  plugin_name: "anomaly-gmm",
  modeltemplates: {
    GaussianMixtureModel: {
      configuration_workflow,
      hyperparameter_fields: ({ table, configuration }) => [
        {
          name: "clusters",
          label: "Clusters",
          type: "Integer",
          attributes: { min: 0 },
        },
      ],
      metrics: { AIC: { lowerIsBetter: true }, BIC: { lowerIsBetter: true } },
      prediction_outputs: [
        { name: "log_likelihood", type: "Float" },
        { name: "cluster", type: "Integer" },
      ],
      train: async ({
        table,
        configuration: { columns },
        hyperparameters,
        state,
      }) => {
        //write data to CSV
        const fields = table.fields;

        readState(state, fields);
        const { joinFields, aggregations } = picked_fields_to_query(
          columns,
          fields
        );
        const where = await stateFieldsToWhere({ fields, state, table });
        let rows = await table.getJoinedRows({
          where,
          joinFields,
          aggregations,
        });

        await write_csv(rows, columns, fields, "/tmp/scdata.csv");

        //run notebook
        await exec(
          `quarto render ${__dirname}/GMM.qmd --to html -o scmodelreport.html`,
          { cwd: "/tmp" }
        );
        //pick up
        const fit_object = await fsp.readFile("/tmp/scanomallymodel");
        const report = await fsp.readFile("/tmp/scmodelreport.html");
        const metric_values = JSON.parse(
          await fsp.readFile("/tmp/scmodelmetrics.json")
        );
        return {
          fit_object,
          report,
          metric_values,
        };
      },
      predict: async ({
        id, //instance id
        model: {
          configuration: { columns },
          table_id,
        },
        hyperparameters,
        fit_object,
        rows,
      }) => {},
    },
    anomaly_gmm: {
      configuration_workflow,
      hyperparameter_fields: ({ table, configuration }) => [
        {
          name: "clusters",
          label: "Clusters",
          type: "Integer",
          attributes: { min: 0 },
        },
      ],
      metrics: { AIC: { lowerIsBetter: true }, BIC: { lowerIsBetter: true } },
      prediction_outputs: [
        { name: "log_likelihood", type: "Float" },
        { name: "cluster", type: "Integer" },
      ],
      train: async ({
        table,
        configuration: { columns },
        hyperparameters,
        state,
      }) => {
        const fields = table.fields;

        readState(state, fields);
        const { joinFields, aggregations } = picked_fields_to_query(
          columns,
          fields
        );
        const where = await stateFieldsToWhere({ fields, state, table });
        let rows = await table.getJoinedRows({
          where,
          joinFields,
          aggregations,
        });

        await write_csv(rows, columns, fields, "/tmp/scdata.csv");

        await python.ex`
        import pickle
        import pandas
        from sklearn.mixture import GaussianMixture
        def train_model():
          df = pandas.read_csv('/tmp/scdata.csv')
          gm = GaussianMixture(n_components=${hyperparameters.clusters}, random_state=0).fit(df)
          with open('/tmp/scanomallymodel', 'wb') as handle:
              pickle.dump(gm, handle, protocol=pickle.HIGHEST_PROTOCOL)
          return {
            'AIC': gm.aic(df),
            'BIC': gm.bic(df)
          }
        `;
        const trainres = await python`train_model()`;
        const blob = await fsp.readFile("/tmp/scanomallymodel");
        return {
          fit_object: blob,
          report: "",
          metric_values: { AIC: trainres.AIC, BIC: trainres.BIC },
        };
      },
      predict: async ({
        id, //instance id
        model: {
          configuration: { columns },
          table_id,
        },
        hyperparameters,
        fit_object,
        rows,
      }) => {
        await fsp.writeFile("/tmp/scanomallymodel" + id, fit_object);
        const table = Table.findOne({ id: table_id });
        const rnd = Math.round(Math.random() * 10000);
        await write_csv(rows, columns, table.fields, `/tmp/scdata${rnd}.csv`);
        await python.ex`
        import pickle
        import pandas
        def gmpredictor(minst_id, filenm):
          with open('/tmp/scanomallymodel'+str(minst_id), "rb") as input_file:
            gm1 = pickle.load(input_file)
            predcsv = pandas.read_csv(filenm)
            return {
               'log_likelihood': list(gm1.score_samples(predcsv)),
               'cluster': list(map(int,gm1.predict(predcsv)))
              }`;
        const predicts =
          await python`gmpredictor(${id}, ${`/tmp/scdata${rnd}.csv`})`;
        return rows.map((r, ix) => ({
          log_likelihood: predicts.log_likelihood[ix],
          cluster: predicts.cluster[ix],
        }));
      },
    },
  },
};
