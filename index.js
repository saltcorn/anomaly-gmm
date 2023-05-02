const Field = require("@saltcorn/data/models/field");
const Table = require("@saltcorn/data/models/table");
const Form = require("@saltcorn/data/models/form");
const Workflow = require("@saltcorn/data/models/workflow");
const FieldRepeat = require("@saltcorn/data/models/fieldrepeat");

const {
  field_picker_fields,
  picked_fields_to_query,
  stateFieldsToWhere,
  stateFieldsToQuery,
  readState,
} = require("@saltcorn/data/plugin-helper");

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

      train: async ({ table, configuration, hyperparameters }) => {
        return { blob: 1, report: "", metric_values: {} };
      },
      predict: async ({ table, configuration, hyperparameters, blob, row }) => {
        return { anomaly: 5 };
      },
    },
  },
};
