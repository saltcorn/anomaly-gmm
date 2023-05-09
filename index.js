const Field = require("@saltcorn/data/models/field");
const Table = require("@saltcorn/data/models/table");
const Form = require("@saltcorn/data/models/form");
const Workflow = require("@saltcorn/data/models/workflow");
const FieldRepeat = require("@saltcorn/data/models/fieldrepeat");
const { eval_expression } = require("@saltcorn/data/models/expression");
const fs = require("fs");
const fsp = fs.promises;
const path = require("path");
const {
  field_picker_fields,
  picked_fields_to_query,
  stateFieldsToWhere,
  stateFieldsToQuery,
  readState,
} = require("@saltcorn/data/plugin-helper");
const { get_predictor, write_csv } = require("@saltcorn/data/model-helper");

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

let gmm_pred_code = get_predictor(path.join(__dirname, "GMM.ipynb"));

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
          `jupyter nbconvert --to html --ClearOutputPreprocessor.enabled=True --embed-images ${__dirname}/GMM.ipynb --execute --output /tmp/scmodelreport.html`,
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
      }) => {
        await fsp.writeFile("/tmp/scanomallymodel" + id, fit_object);
        const table = Table.findOne({ id: table_id });
        const rnd = Math.round(Math.random() * 10000);
        await write_csv(rows, columns, table.fields, `/tmp/scdata${rnd}.csv`);
        await eval("python.ex`" + gmm_pred_code + "`");
        const predicts =
          await python`predict('/tmp/scanomallymodel'+str(${id}), ${`/tmp/scdata${rnd}.csv`})`;
        return rows.map((r, ix) => ({
          log_likelihood: predicts.log_likelihood[ix],
          cluster: predicts.cluster[ix],
        }));
      },
    },
  },
};
