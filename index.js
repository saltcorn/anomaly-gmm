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

let python = pythonBridge({
  python: "python3",
});

const configuration_workflow = (req) =>
  new Workflow({
    steps: [
      {
        name: "Columns",
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

const train_script = ({ data_file, model_file, n_components }) => `
import pickle
import pandas as pd
from sklearn.mixture import GaussianMixture

df = pd.read_csv('${data_file}')
gm = GaussianMixture(n_components=${n_components}, random_state=0).fit(df)
with open('${model_file}', 'wb') as handle:
    pickle.dump(gm, handle, protocol=pickle.HIGHEST_PROTOCOL)
`;

module.exports = {
  sc_plugin_api_version: 1,
  plugin_name: "anomaly-gmm",
  metrics: {},
  modeltemplates: {
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
        const outstream = fs.createWriteStream("/tmp/scdata.csv");
        outstream.write(colWriters.map((cw) => cw.header).join(",") + "\n");
        rows.forEach((row) => {
          outstream.write(
            colWriters.map((cw) => cw.write(row)).join(",") + "\n"
          );
        });
        outstream.end();
        await python.ex(
          train_script({
            data_file: "/tmp/scdata.csv",
            model_file: "/tmp/scanomallymodel",
            n_components: hyperparameters.clusters,
          })
        );
        const blob = await fsp.readFile("/tmp/scanomallymodel");
        return { blob, report: "", metric_values: {} };
      },
      predict: async ({
        table,
        configuration,
        hyperparameters,
        blob,
        rows,
      }) => {
        const result = {};
        rows.forEach((row) => {
          result[row.id] = { anomaly: 5 };
        });
        return result;
      },
    },
  },
};
